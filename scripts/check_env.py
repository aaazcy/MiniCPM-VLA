#!/usr/bin/env python3
"""scripts/check_env.py — environment sanity check before any training/eval run.

Checks: GPU presence, CUDA, torch build, BF16 support, vLLM import, disk free space,
data manifests. Exits non-zero if the machine is not ready for Phase 1.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OK, WARN, FAIL = "✅", "⚠️ ", "❌"
problems = []


def report(name: str, status: str, detail: str = ""):
    print(f"{status} {name:35s} {detail}")
    if status == FAIL:
        problems.append(name)


def main():
    import torch

    print("=" * 72)
    print("VLA-vs-Omni environment check")
    print("=" * 72)

    # ---- GPU / CUDA ----
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        report("GPU", OK, f"{n} x {name} ({mem:.1f} GiB)")
        report("BF16 supported", OK if torch.cuda.is_bf16_supported() else FAIL,
               str(torch.cuda.is_bf16_supported()))
    else:
        report("GPU", FAIL, "no CUDA device — mount a GPU (L40S/A6000 48G recommended, "
                             "RTX4090 24G fallback with group_size=2) before Phase 1")

    report("torch", OK, f"{torch.__version__} (cu{torch.version.cuda})")

    # ---- vLLM ----
    try:
        import vllm
        report("vLLM", OK, f"vLLM {vllm.__version__}")
    except ImportError:
        report("vLLM", WARN, "not installed — pip install vllm==0.6.6.post1 in Phase 1")
    except Exception as e:
        report("vLLM", FAIL, f"import failed: {e}")

    # ---- python deps (CPU-safe) ----
    for mod, pkg in [("yaml", "pyyaml"), ("pydantic", "pydantic"), ("PIL", "pillow"),
                     ("numpy", "numpy"), ("peft", "peft"), ("transformers", "transformers"),
                     ("accelerate", "accelerate"), ("trl", "trl")]:
        try:
            __import__(mod)
            report(pkg, OK)
        except ImportError:
            report(pkg, WARN, "missing (needed before Phase 1 training)")

    # ---- disk ----
    for path, label in [(PROJECT_ROOT, "data disk (project)"), (Path("/"), "system disk")]:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / 1e9
        status = OK if free_gb > 20 else (WARN if free_gb > 5 else FAIL)
        report(f"disk free [{label}]", status, f"{free_gb:.1f} GiB")

    # ---- data manifests ----
    data_dir = PROJECT_ROOT / "data"
    for d in ["calvin/splits", "bridge", "csi"]:
        if (data_dir / d).exists():
            report(f"data present [{d}]", OK)
        else:
            report(f"data present [{d}]", WARN, "run scripts/download_data.sh + preprocess")

    # ---- configs load ----
    try:
        from src.config import load_config
        for g in ["a0", "a1", "b0", "b1", "b2", "b3"]:
            load_config(g)
        report("configs (all groups)", OK)
    except Exception as e:
        report("configs", FAIL, str(e))

    print("=" * 72)
    if problems:
        print(f"NOT READY: {len(problems)} problem(s) -> {', '.join(problems)}")
        sys.exit(1)
    print("READY (or READY for Phase 0 if only GPU/vLLM warnings remain)")


if __name__ == "__main__":
    main()
