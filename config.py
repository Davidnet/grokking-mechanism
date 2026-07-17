import dataclasses


@dataclasses.dataclass(frozen=True)
class GrokkingConfig:
    # task
    p: int = 113
    frac_train: float = 0.3
    seed: int = 0

    # model
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    causal_attention: bool = True

    # derived vocab/sequence sizes: [a, b, "="], "=" token id == p, answer in [0, p)
    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @property
    def n_classes(self) -> int:
        return self.p

    @property
    def seq_len(self) -> int:
        return 3

    # optimizer
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 1.0

    # training
    total_epochs: int = 40_000
    checkpoint_every: int = 100  # -> 400 checkpoints across 40,000 epochs

    def to_json_dict(self) -> dict:
        return dataclasses.asdict(self)
