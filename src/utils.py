import dataclasses
from typing import List, Dict

@dataclasses.dataclass
class opts():
        
    # Data parameters
    root: str = dataclasses.field(default="./data/")
    dataset: str = dataclasses.field(default=None)
    val_seed: int = dataclasses.field(default=0)
    canonical_test_seed: int = dataclasses.field(default=0)
    val_fraction: float = dataclasses.field(default=0.2)
    test_fraction: float = dataclasses.field(default=0.2)
    undirected: bool = dataclasses.field(default=False)
    groundtruth: str = dataclasses.field(default="chipunion")
    split_by_node: bool = dataclasses.field(default=False)
    sampling_power: float = dataclasses.field(default=-0.75)
    
    # Model parameters
    n_conv_layers: int = dataclasses.field(default=None)
    activation_layer: str = dataclasses.field(default="ReLU")
    dropout_ratio: float = dataclasses.field(default=0)
    mplayer: str = dataclasses.field(default=None)
    mplayer_args: List[str] = dataclasses.field(default_factory=list)
    mplayer_kwargs: Dict = dataclasses.field(default_factory=dict)
    latent_dim: int = dataclasses.field(default=None)
    layer_ratio: int = dataclasses.field(default=None)
    decoder: str = dataclasses.field(default=None)
    model: str = dataclasses.field(default="AutoEncoder")
    encoder: str = dataclasses.field(default=None)
    model_path: str = dataclasses.field(default='./models/')
    
    # Training/Evaluation parameters
    lr: float = dataclasses.field(default=0.001)
    es_patience: int = dataclasses.field(default=10)
    val_mode: str = dataclasses.field(default="max")
    val_metric: str = dataclasses.field(default="average_precision_score")
    test_metrics: List[str] = dataclasses.field(default_factory=lambda: ["average_precision_score", "roc_auc_score"])
    n_folds: int = dataclasses.field(default=5)
    epochs: int = dataclasses.field(default=500)
    negative_sampling: str = dataclasses.field(default=None)
    score_batched: bool = dataclasses.field(default=False)
    
    # General settings
    cuda: str = dataclasses.field(default="auto")
    wandb_tracking: bool = dataclasses.field(default=True)
    wandb_project: str = dataclasses.field(default='granpy-dev')
    wandb_save_model: bool = dataclasses.field(default=True)
    wandb_group: str = dataclasses.field(default=None)
    cache_model: bool = dataclasses.field(default=False)    
                

def dataset_hash_keys():
    keys = [
        'val_seed',
        'canonical_test_seed',
        'val_fraction',
        "test_fraction",
        "dataset",
        "undirected",
        "groundtruth",
        "split_by_node",
        "sampling_power"
        ]
    
    return keys


def model_hash_keys():
    keys = dataset_hash_keys()
    keys += [
        "n_conv_layers",
        "activation_layer",
        "dropout_ratio",
        "mplayer",
        "mplayer_args",
        "mplayer_kwargs",
        "latent_dim",
        "layer_ratio",
        "model_path",
        "lr",
        "es_patience",
        "decoder",
        "model",
        "encoder",
        "val_mode",
        "val_metric",
        "test_metrics",
        "epochs",
        "negative_sampling"
    ]

    return keys


def get_dataset_hash(opts):

    include = dataset_hash_keys()
    
    return hash(opts, include)


def get_model_hash(opts):

    include = model_hash_keys()

    return hash(opts, include)


def hash(dataclass_object, include):
    import hashlib
    # inspired by https://death.andgravity.com/stable-hashing

    fields = dataclasses.fields(dataclass_object)

    rv = {}
    for field in fields:
        if field.name in include:
            value = getattr(dataclass_object, field.name)
            if value is None or not value:
                continue
            rv[field.name] = value

    m = hashlib.blake2b(digest_size=5)
    m.update(str(rv).encode("utf-8"))
    return m.hexdigest()
