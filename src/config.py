"""Configuration loader — reads config.yaml and exposes typed access."""
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 768
    d_ff: int = 4096
    n_layers: int = 12
    n_heads: int = 12
    n_experts: int = 8
    top_k: int = 2
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    use_flash_attention: bool = True
    use_grad_checkpoint: bool = True

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        embed = self.vocab_size * self.d_model
        attn_per_layer = 4 * self.d_model * self.d_model
        moe_per_layer = self.n_experts * 3 * self.d_model * self.d_ff
        router_per_layer = self.d_model * self.n_experts
        norm_per_layer = 2 * self.d_model
        total = (
            embed
            + self.n_layers * (attn_per_layer + moe_per_layer + router_per_layer + norm_per_layer)
            + self.d_model
        )
        if not self.tie_embeddings:
            total += embed
        return total


@dataclass
class TrainingConfig:
    batch_size: int = 32
    grad_accum_steps: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 50000
    grad_clip: float = 1.0
    aux_loss_weight: float = 0.01
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 10
    seed: int = 42
    use_bf16: bool = True
    use_wandb: bool = False


@dataclass
class DataConfig:
    dataset: str = "uonlp/CulturaX"
    languages: list = field(default_factory=lambda: ["en", "fr"])
    n_docs_per_lang: int = 750000
    seq_len: int = 1024
    val_ratio: float = 0.01
    tokenizer_train_docs: int = 200000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
        )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
