import dataclasses
from training import TrainingPipeline
from typing import List, Dict


@dataclasses.dataclass
class opts:
    val_seed: int = 2
    canonical_test_seed: int = 1
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    n_conv_layers: int = 1
    activation_layer: str = "ReLU"
    dropout_ratio: float = 0.5
    mplayer: str = "GCNConv"
    mplayer_args: List[str] = dataclasses.field(default_factory=list)
    mplayer_kwargs: Dict = dataclasses.field(default_factory=dict)
    latent_dim: int = 32
    layer_ratio: int = 10
    root: str = "/mnt/storage/granpy/data"
    dataset: str = "jackson"
    model_path: str = "/mnt/storage/granpy/models/"
    lr: float = 1e-3
    es_patience: int = 10
    decoder: str = "MLPDecoder"
    model: str = "AutoEncoder"
    encoder: str = "GAE_Encoder"
    val_mode: str = "max"
    val_metric: str = "average_precision_score"
    test_metrics: List[str] = dataclasses.field(default_factory=list)
    n_folds: int = 5
    epochs: int = 500
    cuda: bool = True
    negative_sampling: str = "structured"
    score_batched: bool = False
    wandb_tracking: bool = True
    wandb_project: str = "granpy-dev"
    wandb_save_model: bool = True
    wandb_group: str = None
    cache_model: bool = False

_opts = opts()

_opts.test_metrics += ["average_precision_score", "roc_auc_score"]

pipeline = TrainingPipeline(_opts)

result = pipeline.run()
print(result)