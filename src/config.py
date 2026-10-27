"""src/config.py — pydantic-validated, frozen experiment configuration.

All hyperparameters are locked here. A run loads `configs/base.yaml`, then merges
the per-group override `configs/group/<group>.yaml`. The resolved config is frozen,
validated, logged to wandb, and stored next to every checkpoint for reproducibility.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


# --------------------------------------------------------------------------- #
# Config schema
# --------------------------------------------------------------------------- #
class ExperimentCfg(BaseModel):
    name: str
    wandb_project: str
    wandb_entity: Optional[str] = None
    seed: int = 42
    output_dir: str
    data_dir: str
    hf_cache_dir: str

    @field_validator("output_dir", "data_dir", "hf_cache_dir")
    @classmethod
    def _abspath(cls, v: str) -> str:
        return str(Path(v).expanduser())


class EnvironmentCfg(BaseModel):
    name: Literal["calvin"]
    max_episode_steps: int = 100
    action_dim: int = 7
    image_size: int = 224
    cameras: List[str] = ["rgb_static"]


class DatasetCfg(BaseModel):
    calvin_root: str
    bridge_root: str
    csi_root: str
    train_ratio: float = 0.85
    bridge_max_trajectories: int = 20000
    bridge_sample_seed: int = 42
    filter_collision: bool = True
    filter_stalled_gripper: bool = True

    @field_validator("train_ratio")
    @classmethod
    def _ratio(cls, v: float) -> float:
        assert 0 < v < 1, "train_ratio must be in (0,1)"
        return v


class SftCfg(BaseModel):
    loss: Literal["mse"] = "mse"
    optimizer: Literal["adamw"] = "adamw"
    lr_action_head: float = 1.0e-4
    lr_lora: float = 2.0e-5
    batch_size: int = 2
    steps_per_traj: int = 2
    val_chunk_size: int = 8       # steps per val item (val uses full-coverage chunks)
    weight_decay: float = 1.0e-4
    gradient_checkpointing: bool = True
    max_epochs: int = 20
    early_stop_patience: int = 3
    bf16: bool = True


class GrpoCfg(BaseModel):
    group_size: int = 4
    rollout_per_group: int = 4
    clip_eps: float = 0.2
    beta_kl: float = 0.05
    lr_action_head: float = 1.0e-4
    lr_lora: float = 2.0e-5
    max_grad_norm: float = 1.0
    optimizer: Literal["adamw"] = "adamw"
    bf16: bool = True
    total_env_steps: int = 400000
    eval_every_episodes: int = 200
    max_traj_len: int = 100


class RewardCfg(BaseModel):
    success: float = 10.0
    lambda_dist: float = 0.5
    lambda_action: float = 0.1
    use_progress_proxy_for_dist: bool = True


class ActionHeadCfg(BaseModel):
    type: Literal["discrete", "continuous"] = "discrete"
    num_bins: int = 256
    hidden_dim: int = 4096
    dims: List[int] = [512, 256]
    action_dim: int = 7
    init_std: float = 0.02


class ModelCfg(BaseModel):
    action_head: ActionHeadCfg = ActionHeadCfg()


class RolloutCfg(BaseModel):
    engine: Literal["vllm", "hf"] = "vllm"
    gpu_memory_utilization: float = 0.85
    temperature: float = 1.0
    max_new_tokens: int = 1
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"


class EvalCfg(BaseModel):
    temperature: float = 0.0
    seeds: int = 10
    max_episodes: int = 34
    id_test: str = "calvin_A_val"
    ood_test: str = "calvin_D"
    robustness: bool = True


class WandbCfg(BaseModel):
    mode: Literal["online", "offline"] = "online"
    log_every_steps: int = 10


class TrainCfg(BaseModel):
    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    freeze_vision: bool = True
    freeze_llm: bool = False


class GroupCfg(BaseModel):
    id: str
    family: Literal["vla", "omni"]
    model_id: str
    train: TrainCfg = TrainCfg()
    sft_only: bool = False
    audio: bool = False
    audio_sample_rate: int = 16000
    audio_only: bool = False   # speech is the ONLY language channel (no instruction text)
    dataset: List[Literal["calvin", "bridge", "csi"]]
    is_ablation: bool = False
    rollout_engine_override: Optional[Literal["vllm", "hf"]] = None


class Config(BaseModel):
    experiment: ExperimentCfg
    environment: EnvironmentCfg
    dataset: DatasetCfg
    sft: SftCfg = SftCfg()
    grpo: GrpoCfg = GrpoCfg()
    reward: RewardCfg = RewardCfg()
    model: ModelCfg = ModelCfg()
    rollout: RolloutCfg = RolloutCfg()
    eval: EvalCfg = EvalCfg()
    wandb: WandbCfg = WandbCfg()
    group: GroupCfg

    @model_validator(mode="after")
    def _validate_ablation(self) -> "Config":
        # B3 is an independent ablation: it must be explicitly marked so results
        # are never pooled into head-to-head A-vs-B comparisons.
        if self.group.is_ablation:
            assert self.group.dataset == ["csi"] or "csi" in self.group.dataset, (
                "ablation group must use the CSI dataset"
            )
        return self

    @property
    def run_dir(self) -> Path:
        d = Path(self.experiment.output_dir) / self.group.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def sft_ckpt_dir(self) -> Path:
        return self.run_dir / "sft"

    @property
    def grpo_ckpt_dir(self) -> Path:
        return self.run_dir / "grpo"

    def to_dict(self) -> Dict[str, Any]:
        """Resolved config as a nested dict (for wandb / ckpt metadata)."""
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_refs(node: Any, root: Dict[str, Any]) -> None:
    """Resolve ${experiment.data_dir} style references (base.yaml uses them),
    including refs embedded in longer strings."""
    if isinstance(node, dict):
        for k, v in node.items():
            _expand_refs(v, root)
            if isinstance(node[k], str):
                node[k] = _REF_RE.sub(lambda m: _lookup(root, m.group(1)), node[k])
    elif isinstance(node, list):
        for v in node:
            _expand_refs(v, root)


_REF_RE = __import__("re").compile(r"\$\{([a-zA-Z0-9_.]+)\}")


def _lookup(root: Dict[str, Any], path: str) -> str:
    cur: Any = root
    for part in path.split("."):
        cur = cur[part]
    return str(cur)


def load_config(group: str, base_path: Path = CONFIG_DIR / "base.yaml") -> Config:
    """Load base config and merge the per-group override, then validate + freeze."""
    group = group.lower()
    with open(base_path) as f:
        base = yaml.safe_load(f)

    override_path = CONFIG_DIR / "group" / f"{group}.yaml"
    if not override_path.exists():
        raise FileNotFoundError(
            f"no group config at {override_path}; available: "
            f"{[p.stem for p in (CONFIG_DIR / 'group').glob('*.yaml')]}"
        )
    with open(override_path) as f:
        override = yaml.safe_load(f)

    # Expand ${experiment.data_dir} style references in base.yaml
    merged = _deep_merge(base, override)
    _expand_refs(merged, merged)

    # env var overrides keep the machine portable (paths, wandb offline)
    if os.environ.get("VLA_DATA_DIR"):
        merged["dataset"]["calvin_root"] = os.environ["VLA_DATA_DIR"] + "/calvin"
        merged["dataset"]["bridge_root"] = os.environ["VLA_DATA_DIR"] + "/bridge"
        merged["dataset"]["csi_root"] = os.environ["VLA_DATA_DIR"] + "/csi"
    if os.environ.get("VLA_OUTPUT_DIR"):
        merged["experiment"]["output_dir"] = os.environ["VLA_OUTPUT_DIR"]
    if os.environ.get("WANDB_MODE"):
        merged["wandb"]["mode"] = os.environ["WANDB_MODE"]

    cfg = Config.model_validate(merged)
    return cfg
