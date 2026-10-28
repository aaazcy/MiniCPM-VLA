"""src/models/openvla.py — OpenVLA-7B wrapper.

Backbone: fused DINOv2+SigLIP vision + Llama2-7B LLM + native discretized action head
(verified against the real modeling code):
  * the "native action head" = the LAST 256 lm_head logits (uniform bins in [-1,1],
    `self.bin_centers`); one action dimension = one generated token per step;
  * `predict_action()` decodes via `discretized = vocab_size - token_ids`.

Common interface (shared with MiniCPM-o wrapper):
    forward_sft(images, lang_texts, audio_wavs=None, actions=None) -> (B,7)
        actions given (SFT): teacher-forced (B,7,256) logits -> E[action] = softmax @
        bin_centers (differentiable, keeps the spec's MSE objective).
        actions None (rollout/eval): autoregressive generate + bin decode.
    logp_actions(images, lang_texts, actions, audio_wavs=None) -> (B,)
        categorical logp of quantized actions under the native head.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from peft import LoraConfig, get_peft_model

log = logging.getLogger(__name__)

EMPTY_TOKEN = 29871  # OpenVLA's special empty token appended after the prompt


class OpenVLAWrapper(nn.Module):
    def __init__(self, model_id: str, freeze_vision: bool = True, use_lora: bool = True,
                 lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.0,
                 hf_cache_dir: Optional[str] = None, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor  # lazy import

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                                       cache_dir=hf_cache_dir)
        self.base = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True, cache_dir=hf_cache_dir
        )
        self.llm = self.base.language_model
        self.vision = self.base.vision_backbone
        self.hidden_dim = 4096
        self.dtype = dtype
        self.vocab_size = self.base.vocab_size
        self.bin_centers = self.base.bin_centers          # (256,) uniform in [-1,1]
        self.num_bins = len(self.bin_centers)
        self.action_dim = 7
        self.action_head_type = "discrete"  # native 256-bin head (CE target)
        try:
            self.base.gradient_checkpointing_enable()
        except Exception as e:
            log.warning("gradient checkpointing not enabled: %s", e)

        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False
        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.llm = get_peft_model(self.llm, lora_cfg)
            self.base.language_model = self.llm
            log.info("OpenVLA: LoRA on LLM (r=%d, alpha=%d)", lora_r, lora_alpha)

        # Full fine-tune the native action head (lm_head) so it can adapt to CALVIN.
        # LoRA only adapts the LLM representation; a frozen pre-trained head collapses
        # to a constant action (smoothness 0) on out-of-domain data. Only the last
        # num_bins columns receive gradients (the CE loss uses just those), so the
        # pre-trained text-vocab columns are effectively untouched.
        self.lm_head = getattr(self.llm, "lm_head", None) or self.llm.base_model.model.lm_head
        for p in self.lm_head.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------ #
    def prepare_for_sft(self) -> None:
        # LoRA params are already requires_grad=True from get_peft_model;
        # do NOT call requires_grad_(True) on the PeftModel (it unfreezes the base).
        assert any(p.requires_grad for p in self.llm.parameters()), \
            "no trainable LoRA parameters"

    def prepare_for_grpo(self, reference_ckpt: Optional[str] = None) -> None:
        self.prepare_for_sft()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ #
    def _quantize(self, actions: torch.Tensor) -> torch.Tensor:
        """(B,7) [-1,1] -> bin indices (B,7) in [0, 255]."""
        q = ((actions.clamp(-1, 1) + 1) / 2 * (self.num_bins - 1)).round().long()
        return q.clamp(0, self.num_bins - 1)

    def quantize(self, actions: torch.Tensor) -> torch.Tensor:
        """Public alias for the SFT trainer (natural bin indices)."""
        return self._quantize(actions)

    def expected_action(self, logits: torch.Tensor) -> torch.Tensor:
        """E[action] under the (natural-order) bin distribution -> (B,7)."""
        probs = torch.softmax(logits.float(), dim=-1)
        centers = torch.from_numpy(np.asarray(self.bin_centers, dtype=np.float32)) \
            .to(logits.device)
        return (probs * centers).sum(-1)

    def _token_ids(self, actions: torch.Tensor) -> torch.Tensor:
        """bin indices -> vocabulary token ids (inverse of predict_action's decode)."""
        return self.vocab_size - self._quantize(actions) - 1

    def _teacher_force_ids(self, inputs: dict, action_tokens: torch.Tensor):
        """prompt (+ EMPTY_TOKEN if absent) + action tokens, with matching mask.

        Matches OpenVLA's training-time convention (predict_action inserts the empty
        token before generating). Without this, SFT/logp are computed on a different
        input than the rollout's generate() path uses.
        """
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        if ids[0, -1].item() != EMPTY_TOKEN:
            ids = torch.cat([ids, torch.full((ids.shape[0], 1), EMPTY_TOKEN,
                                             dtype=ids.dtype, device=ids.device)], dim=1)
            mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype,
                                               device=mask.device)], dim=1)
        ids = torch.cat([ids, action_tokens], dim=1)
        mask = torch.cat([mask, torch.ones(action_tokens.shape, dtype=mask.dtype,
                                           device=action_tokens.device)], dim=1)
        return ids, mask

    def _prep_inputs(self, images: torch.Tensor, lang_texts: List[str]):
        from PIL import Image
        imgs = [im.detach().cpu().float().numpy() for im in images]
        if imgs[0].max() <= 1.5:
            imgs = [(im * 255).astype("uint8") for im in imgs]
        imgs = [Image.fromarray(im.transpose(1, 2, 0)) for im in imgs]
        inputs = self.processor(text=list(lang_texts), images=imgs,
                                padding=True, truncation=True, max_length=64,
                                return_tensors="pt").to(images.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        return inputs

    def _head_logits(self, inputs: dict, ids: torch.Tensor, mask: torch.Tensor,
                     n_pos: int) -> torch.Tensor:
        """Forward through the LLM, then apply the lm_head ONLY to the last n_pos
        positions (avoids the (B, T, vocab) logits blowup that OOMs 32 GB cards)."""
        out = self.base(input_ids=ids, attention_mask=mask,
                        pixel_values=inputs["pixel_values"],
                        output_hidden_states=True)
        # Causal LM: hidden_state[p] predicts token[p+1]. The 7 action tokens are
        # appended at the END of `ids` (positions T..T+6), so the hidden states that
        # PREDICT them are positions T-1..T+5 = [-n_pos-1 : -1] (NOT [-n_pos:], which
        # are the positions AT the action tokens and predict one step past them).
        hs = out.hidden_states[-1][:, -n_pos - 1:-1, :]      # (B, n_pos, H)
        lm_head = getattr(self.llm, "lm_head", None) or self.llm.base_model.model.lm_head
        logits_full = lm_head(hs)                               # (B, n_pos, V)
        # The last num_bins vocab columns hold the action tokens in REVERSED bin
        # order (token vocab_size-1 == bin 0 == -1). Flip so slice index j == bin j,
        # matching bin_centers[j] and _quantize's output.
        return logits_full[:, :, -self.num_bins:].flip(-1)      # (B, n_pos, num_bins)

    def forward(self, images: torch.Tensor, lang_texts: List[str],
                audio_wavs: Optional[List] = None,
                actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        # alias so DDP hooks apply: DDP intercepts .forward(), NOT .forward_sft
        return self.forward_sft(images, lang_texts, audio_wavs, actions)

    def forward_sft(self, images: torch.Tensor, lang_texts: List[str],
                    audio_wavs: Optional[List] = None,
                    actions: Optional[torch.Tensor] = None,
                    temperature: float = 1.0) -> torch.Tensor:
        """Predicted 7-D action in [-1,1].

        With `actions` (SFT): returns teacher-forced native-head logits
        (B,7,num_bins) in natural bin order; the trainer computes cross-entropy
        on the quantized bins. Without (rollout/eval): autoregressive generation
        through the native head (temperature=0 -> greedy/deterministic).
        """
        if actions is None:
            return self._generate_action(images, lang_texts, temperature)

        inputs = self._prep_inputs(images, lang_texts)
        tok = self._token_ids(actions.to(images.device))
        ids, mask = self._teacher_force_ids(inputs, tok)
        logits = self._head_logits(inputs, ids, mask, self.action_dim)  # (B,7,num_bins)
        return logits  # trainer computes cross-entropy on the quantized bins

    def sft_loss(self, images: torch.Tensor, lang_texts: List[str],
                 audio_wavs: Optional[List], actions: torch.Tensor):
        """OpenVLA SFT objective: cross-entropy on the discretized action bins."""
        logits = self.forward_sft(images, lang_texts, audio_wavs, actions)
        q = self._quantize(actions.to(logits.device))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), q.reshape(-1))
        with torch.no_grad():
            mse = F.mse_loss(self.expected_action(logits), actions.to(logits.device))
        return {"loss": loss, "mse": mse.detach()}

    @torch.no_grad()
    def _generate_action(self, images: torch.Tensor, lang_texts: List[str],
                         temperature: float = 1.0) -> torch.Tensor:
        """Autoregressive action generation (batch 1); bin decode, no unnormalization."""
        inputs = self._prep_inputs(images, lang_texts)
        ids = inputs["input_ids"]
        if ids[0, -1].item() != EMPTY_TOKEN:
            ids = torch.cat([ids, torch.tensor([[EMPTY_TOKEN]], device=ids.device)], dim=1)
        gen_kwargs = dict(attention_mask=inputs["attention_mask"], max_new_tokens=7)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)
        gen = self.base.generate(ids, **gen_kwargs)
        tokens = gen[0, -self.action_dim:]
        bins = self.vocab_size - tokens.cpu().numpy().astype(np.int64)
        bins = np.clip(bins - 1, 0, self.num_bins - 1)
        return torch.from_numpy(self.bin_centers[bins]).float().unsqueeze(0).to(images.device)

    def logp_actions(self, images: torch.Tensor, lang_texts: List[str],
                     actions: torch.Tensor,
                     audio_wavs: Optional[List] = None) -> torch.Tensor:
        """Categorical logp of the quantized actions under the native head (GRPO)."""
        inputs = self._prep_inputs(images, lang_texts)
        tok = self._token_ids(actions.to(images.device))
        ids, mask = self._teacher_force_ids(inputs, tok)
        logits = self._head_logits(inputs, ids, mask, self.action_dim)  # (B,7,256)
        logp = torch.log_softmax(logits, dim=-1)
        q = self._quantize(actions.to(images.device))
        return logp.gather(-1, q.unsqueeze(-1)).squeeze(-1).sum(-1)  # (B,)

    @torch.no_grad()
    def sample_action(self, images: torch.Tensor, lang_texts: List[str],
                      temperature: float = 1.0, audio_wavs: Optional[List] = None):
        """Sampled action + its logp under the SAMPLING distribution (temp-scaled),
        for GRPO rollout (logp_old) and stochastic action selection."""
        inputs = self._prep_inputs(images, lang_texts)
        ids = inputs["input_ids"]
        if ids[0, -1].item() != EMPTY_TOKEN:
            ids = torch.cat([ids, torch.tensor([[EMPTY_TOKEN]], device=ids.device)], dim=1)
        gen_kwargs = dict(attention_mask=inputs["attention_mask"], max_new_tokens=7)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)
        gen = self.base.generate(ids, **gen_kwargs)
        tokens = gen[0, -self.action_dim:]
        bins = self.vocab_size - tokens.cpu().numpy().astype(np.int64)
        bins = np.clip(bins - 1, 0, self.num_bins - 1)
        action = torch.from_numpy(self.bin_centers[bins]).float().unsqueeze(0) \
            .to(images.device)
        # logp under the sampling distribution (same temperature as generate)
        # Re-encode the DECODED action (not the raw generated tokens): when the model
        # emits a non-action token, `tokens` != `_token_ids(action)`, and using `tokens`
        # here makes `logp_old` inconsistent with `logp_actions` (which re-encodes from
        # the float action). That mismatch explodes the GRPO ratio exp(pi - old).
        tok2 = self._token_ids(action)
        ids2, mask2 = self._teacher_force_ids(inputs, tok2)
        logits = self._head_logits(inputs, ids2, mask2, self.action_dim)
        logp = torch.log_softmax(logits / max(temperature, 1e-6), dim=-1)
        q = torch.from_numpy(bins).long().to(images.device).unsqueeze(0)
        logp_old = logp.gather(-1, q.unsqueeze(-1)).squeeze(-1).sum(-1).item()
        return action, logp_old

    # ------------------------------------------------------------------ #
    def save_adapter(self, path: str) -> None:
        import os
        from peft import PeftModel
        os.makedirs(path, exist_ok=True)
        if isinstance(self.llm, PeftModel):
            self.llm.save_pretrained(path)
        # the full-fine-tuned action head is not part of the LoRA adapter -> save it
        torch.save(self.lm_head.state_dict(), f"{path}/lm_head.pt")

    def load_adapter(self, path: str) -> None:
        import os
        from peft import PeftModel
        if isinstance(self.llm, PeftModel):
            self.llm.load_adapter(path, adapter_name="default")
        head_path = f"{path}/lm_head.pt"
        if os.path.exists(head_path):
            self.lm_head.load_state_dict(torch.load(head_path, map_location="cpu"))
