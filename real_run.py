from src.utils import opts
from training import TrainingPipeline

_opts = opts(
    # Data parameters
    dataset= "shalek",
    val_seed= 12,
    canonical_test_seed= 1,
    val_fraction= 0.2,
    test_fraction= 0.2,
    undirected= False,
    
    # Model parameters
    n_conv_layers= 2,
    activation_layer= "ReLU",
    dropout_ratio= 0.5,
    mplayer= "GCNConv",
    mplayer_args= [],
    mplayer_kwargs= dict(),
    latent_dim= 32,
    layer_ratio= 10,
    decoder= "MLPDecoder",
    model= "AutoEncoder",
    encoder= "GNNEncoder",
    model_path= "./models/",
    
    # Training/Evaluation parameters
    lr= 1e-3,
    es_patience= 10,
    val_mode= "max",
    val_metric= "average_precision_score",
    test_metrics= ["average_precision_score", "roc_auc_score"],
    n_folds= 5,
    epochs= 500,
    negative_sampling= "structured_tail",
    score_batched= False,
    
    #General settings
    cuda= False,
    cache_model = True
)

pipeline = TrainingPipeline(_opts)

result = pipeline.run()
print(result)