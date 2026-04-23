from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    data_path = "en-fr.csv"
    dataset_fraction = 0.002
    dataset_size = None
    max_len = 50
    src_vocab_size = 20000
    tgt_vocab_size = 15000
    reverse_source = True

    embed_dim = 256
    hidden_dim = 256
    n_layers = 2
    dropout = 0.0

    batch_size = 128
    learning_rate = 0.7
    optimizer = "sgd"
    grad_clip = 5.0
    epochs = 10
    use_mixed_precision = True
    lr_decay_start_epoch = 5
    lr_decay_factor = 0.5

    max_decode_len = 60

    param_init_range = 0.08

    seed = 10701

    checkpoint_dir = "checkpoints"

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    cache_path = "data_cache.pt"

    def get_effective_dataset_size(self, total_rows=22_500_000):
        if self.dataset_size is not None:
            return min(self.dataset_size, total_rows)
        return int(self.dataset_fraction * total_rows)

    def scale_vocab_for_fraction(self):
        if self.dataset_fraction <= 0.25:
            self.src_vocab_size = 50000
            self.tgt_vocab_size = 30000
        else:
            self.src_vocab_size = 160000
            self.tgt_vocab_size = 80000


def get_config():
    return Config(
        dataset_fraction=0.25,
        embed_dim=512,
        hidden_dim=512,
        n_layers=4,
        batch_size=32,
        src_vocab_size=30000,
        tgt_vocab_size=20000,
        cache_path=None,
    )
