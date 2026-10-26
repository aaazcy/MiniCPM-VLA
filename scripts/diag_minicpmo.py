#!/usr/bin/env python3
"""Diagnose MiniCPM-o VRAM: patch counts, per-stage memory, gc/sdpa status."""
import sys, argparse
sys.path.insert(0, "/root/autodl-tmp/vla-omni-bench")
import torch
from src.config import load_config
from src.models.factory import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:1")
ap.add_argument("--group", default="b2")
args = ap.parse_args()

cfg = load_config(args.group)
model = build_model(cfg, stage="sft").to(args.device)
print("gc:", model.base.is_gradient_checkpointing if hasattr(model.base, "is_gradient_checkpointing") else "n/a",
      "| llm gc:", model.llm.is_gradient_checkpointing)
print("llm attn impl:", getattr(model.llm.config, "_attn_implementation", None) if hasattr(model.llm, "config") else "n/a")
print("query_num:", getattr(model.base.config, "query_num", None))
print("vision hidden:", getattr(model.base.config.vision_config, "hidden_size", None))
print("slice_mode:", getattr(model.base.config, "slice_mode", None))

images = torch.randint(0, 255, (1, 3, 224, 224), device=args.device).float() / 255.0
langs = ["open the drawer"]
torch.cuda.reset_peak_memory_stats()
inputs = model._prepare_inputs(images, langs, None)
print("input_ids:", tuple(inputs["input_ids"].shape))
print("tgt_sizes:", inputs.get("tgt_sizes"))
print("pixel:", tuple(inputs["pixel_values"].shape) if inputs.get("pixel_values") is not None else None)
print("after processor:", torch.cuda.max_memory_allocated()/1e9, "GB")

embeds, vis = model.base.get_vllm_embedding(inputs)
print("embeds:", tuple(embeds.shape), "| after vllm:", torch.cuda.max_memory_allocated()/1e9, "GB")
embeds2 = model.base.get_omni_embedding(inputs, input_embeddings=embeds,
                                        chunk_length=model.base.config.audio_chunk_length)
print("after omni:", torch.cuda.max_memory_allocated()/1e9, "GB")
out = model.llm(input_ids=None, position_ids=inputs["position_ids"].long(),
                inputs_embeds=embeds2, attention_mask=inputs["attention_mask"],
                output_hidden_states=True)
print("after llm fwd:", torch.cuda.max_memory_allocated()/1e9, "GB")
loss = model.action_head(out.hidden_states[-1][:, -1]).pow(2).mean()
loss.backward()
print("after bwd:", torch.cuda.max_memory_allocated()/1e9, "GB")
print("DIAG DONE")
