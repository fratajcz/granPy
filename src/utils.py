import dataclasses


@dataclasses.dataclass
class opts:
    # TODO: need to check what we need
    root: str = './'
    norm_path: str = './'
    grn_path: str = './'

    norm_file: str = 'gasch_GSE102475.csv'
    grn_file: str = 'yeast_KDUnion.txt'

    seed: int = 45
    use_cuda: bool = False
    wandb: bool = False

    no_features: bool = False
    log_features: bool = True
    cells_min_genes: int = 200  # e.g. 200
    normalize_library: bool = False
    genes_unit_variance: bool = True
    scaler: str = 'None'  # MinMaxScaler, StandardScaler or None

    add_eye: bool = False
    add_metacells: bool = False
    add_binary_feat: bool = False
    add_random_feat: int = 1000
    add_stats_feat: bool = True
    add_node2vec: bool = False
    add_TF_id: bool = True

    genes_min_cells: int = 3  # e.g. 3
    only_HVG: int = 500  # e.g.500
    remove_isolated_genes: bool = True
    remove_isolated_train_genes: bool = False

    val_ratio: float = 0.5
    test_ratio: float = 0.5
    fixed_ground_truth_ratio: float = 0

    epochs: int = 200
    early_stopping: int = 10
    es_metric: str = 'map_tf'  # early stopping metric: ['aupr', 'map_tf', 'map_pot_net']
    learning_rate: float = 0.0001
    weight_decay: float = 0.1758358134292286  # L2 regularization
    lambda_l1: float = 0.3386477958091192  # L1 regularization
    dropout_rate: float = 0.3025696329167428

    train_edge_dropout: float = 0.1939705451217022
    sampling: str = 'pot_net_full'  # for backprop and early stopping: ['random', 'structured', 'pot_net', 'pot_net_full']

    layers: int = 1  # [1, 2, 3]
    latent_dim: int = 87
    layer_ratio: float = 9.126499113087188  # >1
    decoder: str = 'InnerProductDecoder'  # CosineDecoder, InnerProductDecoder, PNormDecoder
    p: int = 2  # relevant for PNormDecoder

    def __init__(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


def dataset_hash_keys():
    keys = [
        'val_seed',
        'canonical_test_seed',
        'val_fraction',
        "test_fraction",
        "root",
        "dataset"
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
