"""src/models/minicpmo.py — MiniCPM-o 4.5 (9B) wrapper for robot manipulation.

Backbone: SigLip2 vision + Whisper audio + Qwen3-8B LLM (verified against the real
`openbmb/MiniCPM-o-4_5` modeling code):
    MiniCPMO.forward(data)          = llm(inputs_embeds=get_omni_embedding(
                                        get_vllm_embedding(data), ...))
    processor(text=, images=, audios=, sampling_rate=16000)
        -> {input_ids, attention_mask, pixel_values, tgt_sizes, image_bound,
            audio_features, audio_feature_lens, audio_bounds, spk_bounds}

Per spec:
  * TTS output branch is never invoked (removed from the training graph);
  * new shared ActionHead appended on the LLM last-token hidden state;
  * B0/B1: backbone frozen, only the action head trains;
  * B2/B3: LoRA (r=8, alpha=16) on the LLM + action head;
  * B3: Whisper audio enabled, synchronized 16 kHz synthesized speech.

Common interface (shared with OpenVLA wrapper):
    forward_sft(images, lang_texts, audio_wavs=None) -> (B, 7)
    logp_actions(images, lang_texts, actions, audio_wavs=None) -> (B,)
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
from torch import nn
from peft import LoraConfig, get_peft_model

from .action_head import ActionHead

log = logging.getLogger(__name__)


class MiniCPMoWrapper(nn.Module):
    def __init__(self, model_id: str, use_lora: bool = True, lora_r: int = 8,
                 lora_alpha: int = 16, lora_dropout: float = 0.0,
                 enable_audio: bool = False, audio_sample_rate: int = 16000,
                 hf_cache_dir: Optional[str] = None,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import AutoModel, AutoProcessor  # lazy import

        self.model_id = model_id
        self.enable_audio = enable_audio
        self.audio_sample_rate = audio_sample_rate
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                                       cache_dir=hf_cache_dir)
        self.base = AutoModel.from_pretrained(model_id, torch_dtype=dtype,
                                              trust_remote_code=True, cache_dir=hf_cache_dir,
                                              attn_implementation="sdpa")
        self.llm = self.base.llm          # Qwen3
        self.hidden_dim = 4096
        self.dtype = dtype
        try:
            self.base.gradient_checkpointing_enable()
            if hasattr(self.base, "llm") and hasattr(self.base.llm, "gradient_checkpointing_enable"):
                self.base.llm.gradient_checkpointing_enable()
        except Exception as e:
            log.warning("gradient checkpointing not enabled: %s", e)

        # freeze the whole backbone; LoRA (if enabled) re-enables the LLM adapters
        for p in self.base.parameters():
            p.requires_grad = False
        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.llm = get_peft_model(self.llm, lora_cfg)
            self.base.llm = self.llm
            # MiniCPM-o's internal code accesses `self.llm.model.embed_tokens`, but
            # through PeftModel `.model` resolves to Qwen3ForCausalLM (which lacks the
            # attr). Register an alias on the underlying causal LM so the original
            # access path keeps working with shared weights.
            qwen_clm = self.llm.base_model.model          # Qwen3ForCausalLM
            qwen_clm.embed_tokens = qwen_clm.model.embed_tokens
            log.info("MiniCPM-o: LoRA on LLM (r=%d, alpha=%d)", lora_r, lora_alpha)

        # new shared action head (spec-exact): Linear(4096,512)->LN->GELU->...->Tanh
        self.action_head = ActionHead(hidden_dim=4096, dims=(512, 256), action_dim=7)

    # ------------------------------------------------------------------ #
    def prepare_for_sft(self) -> None:
        # action head trains; LoRA params are already requires_grad=True from
        # get_peft_model. Do NOT call requires_grad_(True) on the PeftModel
        # (it would unfreeze the whole base).
        self.action_head.requires_grad_(True)

    def prepare_for_grpo(self, reference_ckpt: Optional[str] = None) -> None:
        self.prepare_for_sft()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ #
    def _prepare_inputs(self, images: torch.Tensor, lang_texts: List[str],
                        audio_wavs: Optional[List] = None) -> dict:
        from PIL import Image
        imgs = [im.detach().cpu().float().numpy() for im in images]
        if imgs[0].max() <= 1.5:
            imgs = [(im * 255).astype("uint8") for im in imgs]
        # MiniCPM-o's processor expects PIL images (numpy gets misread as a
        # different aspect ratio and tiles into ~14 slices -> OOM)
        imgs = [Image.fromarray(im.transpose(1, 2, 0)) for im in imgs]

        audios = None
        if self.enable_audio and audio_wavs is not None and all(w is not None for w in audio_wavs):
            audios = [w.detach().cpu().numpy().astype("float32") for w in audio_wavs]

        inputs = self.processor(text=list(lang_texts), images=imgs, audios=audios,
                                sampling_rate=self.audio_sample_rate,
                                max_slice_nums=1,   # 1 tile: ~1024 vision tokens
                                return_tensors="pt").to(images.device)
        bs, seq_len = inputs["input_ids"].shape
        inputs["position_ids"] = torch.arange(seq_len, device=images.device) \
            .unsqueeze(0).expand(bs, -1)
        return inputs

    def _last_token_hidden(self, inputs: dict) -> torch.Tensor:
        # NOTE: we do NOT call base.get_vllm_embedding() here. Its internal
        # scatter_ / += on the embedding tensor is incompatible with gradient
        # checkpointing (in-place op on a leaf view -> RuntimeError during SFT).
        # Reimplement the multimodal embedding fusion without in-place ops.
        scale_emb = getattr(self.llm.config, "scale_emb", None)
        text_embeds = self.llm.model.embed_tokens(inputs["input_ids"])
        if scale_emb is not None:
            text_embeds = text_embeds * scale_emb

        if inputs.get("pixel_values") is not None and len(inputs["pixel_values"]) > 0:
            vision_hidden = self.base.get_vision_embedding(inputs)  # list per sample
            if len(vision_hidden) != len(inputs["input_ids"]):
                log.warning("vision_hidden len %d != batch %d (fallback: text-only for "
                            "missing samples)", len(vision_hidden), len(inputs["input_ids"]))
            injected = []
            for i in range(len(inputs["input_ids"])):
                cur = text_embeds[i]
                if i >= len(vision_hidden):
                    injected.append(cur)
                    continue
                vs = vision_hidden[i]
                bound = inputs["image_bound"][i]
                if len(vs) == 0 or len(bound) == 0:
                    injected.append(cur)
                    continue
                cur2 = cur.clone()  # fresh non-leaf buffer
                indices = torch.stack(
                    [torch.arange(r[0], r[1], dtype=torch.long, device=cur.device)
                     for r in bound])
                cur2[indices.reshape(-1)] = vs.view(-1, vs.shape[-1])
                injected.append(cur2)
            text_embeds = torch.stack(injected)

        embeds = self.base.get_omni_embedding(
            inputs, input_embeddings=text_embeds,
            chunk_length=self.base.config.audio_chunk_length)
        out = self.llm(
            input_ids=None,
            position_ids=inputs["position_ids"].long(),
            inputs_embeds=embeds,
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        hs = out.hidden_states[-1]                       # (B, T, H)
        seq_lens = inputs["attention_mask"].sum(dim=1) - 1
        return hs[torch.arange(len(seq_lens)), seq_lens]

    def forward_sft(self, images: torch.Tensor, lang_texts: List[str],
                    audio_wavs: Optional[List] = None,
                    actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        inputs = self._prepare_inputs(images, lang_texts, audio_wavs)
        return self.action_head(self._last_token_hidden(inputs))

    def logp_actions(self, images: torch.Tensor, lang_texts: List[str],
                     actions: torch.Tensor,
                     audio_wavs: Optional[List] = None) -> torch.Tensor:
        """Gaussian log-likelihood of the continuous actions under the action head
        (unit variance). Used for GRPO importance ratios / KL."""
        mu = self.forward_sft(images, lang_texts, audio_wavs)
        return -0.5 * ((actions - mu) ** 2).sum(-1) - 0.5 * 7 * torch.log(torch.tensor(6.2832))

    # ------------------------------------------------------------------ #
    def save_adapter(self, path: str) -> None:
        import os
        from peft import PeftModel
        os.makedirs(path, exist_ok=True)
        if isinstance(self.llm, PeftModel):
            self.llm.save_pretrained(path)
        torch.save(self.action_head.state_dict(), f"{path}/action_head.pt")

    def load_adapter(self, path: str) -> None:
        from peft import PeftModel
        if isinstance(self.llm, PeftModel):
            self.llm.load_adapter(path, adapter_name="default")
        self.action_head.load_state_dict(torch.load(f"{path}/action_head.pt", map_location="cpu"))
