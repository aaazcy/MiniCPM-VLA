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
import torch.nn.functional as F
import numpy as np
from torch import nn
from peft import LoraConfig, get_peft_model

from .action_head import ActionHead, DiscreteActionHead

log = logging.getLogger(__name__)


class MiniCPMoWrapper(nn.Module):
    def __init__(self, model_id: str, use_lora: bool = True, lora_r: int = 8,
                 lora_alpha: int = 16, lora_dropout: float = 0.0,
                 enable_audio: bool = False, audio_sample_rate: int = 16000,
                 audio_only: bool = False,
                 action_head_type: str = "discrete", num_bins: int = 256,
                 hf_cache_dir: Optional[str] = None,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import AutoModel, AutoProcessor  # lazy import

        self.model_id = model_id
        self.enable_audio = enable_audio
        self.audio_sample_rate = audio_sample_rate
        self.audio_only = audio_only
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

        # new action head (discrete by default, matching OpenVLA's bin parameterization)
        self.action_head_type = action_head_type
        if action_head_type == "discrete":
            self.action_head = DiscreteActionHead(hidden_dim=self.hidden_dim,
                                                  dims=(512, 256), action_dim=7,
                                                  num_bins=num_bins)
        else:
            self.action_head = ActionHead(hidden_dim=self.hidden_dim, dims=(512, 256),
                                          action_dim=7)

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
        imgs = [Image.fromarray(im.transpose(1, 2, 0)) for im in imgs]

        audios = None
        if self.enable_audio and audio_wavs is not None and all(w is not None for w in audio_wavs):
            def _to_wav(w):   # accepts torch.Tensor / np.ndarray / wrapped list
                if isinstance(w, (list, tuple)) and len(w) == 1:
                    w = w[0]
                if hasattr(w, "detach"):
                    w = w.detach().cpu().numpy()
                return np.asarray(w, dtype="float32")
            audios = [_to_wav(w) for w in audio_wavs]
        if self.audio_only and audios is None:
            raise RuntimeError("audio_only=True but no audio wavs provided (B3 speech-only "
                               "input requires a wav for every sample)")

        # The processor requires explicit multimodal placeholders in the text.
        # audio_only (B3): the instruction words are NOT in the text — speech is the
        # ONLY language channel, so text carries just the placeholders.
        if self.audio_only:
            texts = [f"(<image>./</image>)\n(<audio>./</audio>)" for _ in lang_texts]
        else:
            texts = [f"(<image>./</image>)\n{t}" for t in lang_texts]
            if audios is not None:
                texts = [f"(<audio>./</audio>)\n{t}" for t in texts]

        # Wrap in MiniCPM-o's official chat template (matches model.chat / the model's
        # pretraining format): "<|im_start|>user\n{content}<|im_end|>\n
        # <|im_start|>assistant\n". Without it the model sees raw placeholders +
        # instruction and runs in a non-official prompt regime (analogous to OpenVLA's
        # missing "In: ... Out:" wrapper).
        texts = [f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n" for t in texts]

        # The image processor treats a LIST of images as TILES of ONE sample, so
        # batching must be done per-sample and merged manually.
        per = []
        for i in range(len(images)):
            a = [audios[i]] if audios is not None else None
            per.append(self.processor(text=[texts[i]], images=[imgs[i]], audios=a,
                                      sampling_rate=self.audio_sample_rate,
                                      max_slice_nums=1, return_tensors="pt"))

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", 0) or 0
        max_len = max(p["input_ids"].shape[1] for p in per)
        input_ids_list, masks, pixel_values, image_sizes, tgt_sizes, bounds = [], [], [], [], [], []
        for p in per:
            L = p["input_ids"].shape[1]
            pad = max_len - L
            input_ids_list.append(torch.cat([torch.full((1, pad), pad_id, dtype=torch.long),
                                             p["input_ids"]], dim=1))
            masks.append(torch.cat([torch.zeros((1, pad), dtype=torch.bool),
                                    p["attention_mask"].bool()], dim=1))
            pixel_values.extend(p["pixel_values"])          # each is a sample's tiles
            image_sizes.extend(p["image_sizes"])
            ts = p["tgt_sizes"]
            tgt_sizes.append(ts[0] if isinstance(ts, list) else ts)  # (n, 2) tensor
            b = p["image_bound"]
            bounds.append((b[0] if isinstance(b, list) else b) + pad)  # (n,2) + pad offset

        inputs = {
            "input_ids": torch.cat(input_ids_list, dim=0).to(images.device),
            "attention_mask": torch.cat(masks, dim=0).to(images.device),
            "pixel_values": [[t.to(images.device) for t in s] if isinstance(s, (list, tuple))
                             else s.to(images.device) for s in pixel_values],
            "image_sizes": image_sizes,
            "image_bound": [b.to(images.device) for b in bounds],
            "tgt_sizes": torch.cat(tgt_sizes, dim=0).to(images.device),
        }
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
        # Unwrap to the raw Qwen3Model: self.llm may be PeftModelForCausalLM
        # (LoRA) -> Qwen3ForCausalLM -> Qwen3Model. Only Qwen3Model exposes
        # last_hidden_state; the ForCausalLM wrapper's out[0] is LOGITS (B,T,vocab).
        m = self.llm
        while hasattr(m, "model") and type(m).__name__ != "Qwen3Model":
            m = m.model
        out = m(
            input_ids=None,
            position_ids=inputs["position_ids"].long(),
            inputs_embeds=embeds,
            attention_mask=inputs["attention_mask"],
            output_hidden_states=False,
        )
        hs = getattr(out, "last_hidden_state", None)
        if hs is None:
            hs = out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None \
                else out[0]
        seq_lens = inputs["attention_mask"].sum(dim=1) - 1
        return hs[torch.arange(len(seq_lens)), seq_lens]

    def forward(self, images: torch.Tensor, lang_texts: List[str],
                audio_wavs: Optional[List] = None,
                actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        # alias so DDP hooks apply: DDP intercepts .forward(), NOT .forward_sft
        return self.forward_sft(images, lang_texts, audio_wavs, actions)

    def forward_sft(self, images: torch.Tensor, lang_texts: List[str],
                    audio_wavs: Optional[List] = None,
                    actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        inputs = self._prepare_inputs(images, lang_texts, audio_wavs)
        h = self._last_token_hidden(inputs)
        if self.action_head_type == "discrete":
            logits = self.action_head(h)
            if actions is None:
                return self.action_head.decode(logits, temperature=0.0)
            # return the raw logits (B, 7, num_bins) so the trainer can compute
            # cross-entropy on the quantized bins; returning expected_action here
            # would silently degrade the discrete head to MSE regression.
            return logits
        return self.action_head(h)

    def quantize(self, actions: torch.Tensor) -> torch.Tensor:
        """Public alias for the SFT trainer (natural bin indices)."""
        return self.action_head.quantize(actions)

    def expected_action(self, logits: torch.Tensor) -> torch.Tensor:
        """E[action] under the bin distribution -> (B,7)."""
        return self.action_head.expected_action(logits)

    def sft_loss(self, images: torch.Tensor, lang_texts: List[str],
                 audio_wavs: Optional[List], actions: torch.Tensor):
        """SFT objective -> {'loss': optimization target, 'mse': action MSE}.

        discrete: cross-entropy on quantized bins (loss) + expected-action MSE (metric).
        continuous: MSE (loss == mse)."""
        inputs = self._prepare_inputs(images, lang_texts, audio_wavs)
        h = self._last_token_hidden(inputs)
        actions = actions.to(h.device).float()
        if self.action_head_type == "discrete":
            logits = self.action_head(h)
            q = self.action_head.quantize(actions)
            loss = F.cross_entropy(logits.reshape(-1, self.action_head.num_bins),
                                   q.reshape(-1))
            mse = F.mse_loss(self.action_head.expected_action(logits), actions)
            return {"loss": loss, "mse": mse.detach()}
        pred = self.action_head(h)
        loss = F.mse_loss(pred, actions)
        return {"loss": loss, "mse": loss.detach()}

    @torch.no_grad()
    def sample_action(self, images: torch.Tensor, lang_texts: List[str],
                      temperature: float = 1.0,
                      audio_wavs: Optional[List] = None):
        """Sample an action + its logp under the sampling distribution (rollout)."""
        inputs = self._prepare_inputs(images, lang_texts, audio_wavs)
        h = self._last_token_hidden(inputs)
        if self.action_head_type == "discrete":
            logits = self.action_head(h)
            return self.action_head.sample(logits, temperature)
        mu = self.action_head(h)
        if temperature > 0:
            noise = torch.randn_like(mu) * (0.1 * temperature)
            action = (mu + noise).clamp(-1, 1)
            logp = -0.5 * ((action - mu) ** 2).sum(-1) / (0.1 ** 2)
        else:
            action, logp = mu, torch.zeros(mu.shape[0], device=mu.device)
        return action, logp

    def logp_actions(self, images: torch.Tensor, lang_texts: List[str],
                     actions: torch.Tensor,
                     audio_wavs: Optional[List] = None) -> torch.Tensor:
        """Log-likelihood of the actions under the action head (GRPO ratio).

        discrete: categorical logp over bins; continuous: Gaussian (sigma=0.1)."""
        inputs = self._prepare_inputs(images, lang_texts, audio_wavs)
        h = self._last_token_hidden(inputs)
        if self.action_head_type == "discrete":
            logits = self.action_head(h)
            return self.action_head.logp(logits, actions)
        mu = self.action_head(h)
        var = 0.1 ** 2
        return -0.5 * ((actions - mu) ** 2).sum(-1) / var \
            - 0.5 * 7 * torch.log(torch.tensor(6.2832 * var, device=mu.device,
                                               dtype=mu.dtype))

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
