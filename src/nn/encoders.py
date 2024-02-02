import torch
from src.nn.nn_utils import get_layer, get_mp_layer
from torch_geometric.nn import Sequential

class GAE_Encoder(torch.nn.Module):
    def __init__(self, input_dim, opts):
        assert opts.n_conv_layers in [1, 2, 3]
        assert opts.layer_ratio >= 1

        super(GAE_Encoder, self).__init__()
        self.n_conv_layers = opts.n_conv_layers
        hidden1 = int(opts.latent_dim*pow(opts.layer_ratio, opts.n_conv_layers-1))

        layers = torch.nn.ModuleList()
        flow = []

        layers.append(get_layer("Dropout")(opts.dropout_ratio))
        flow.append("x -> x")

        layers.append(get_mp_layer(opts.mplayer)(input_dim, hidden1, *opts.mplayer_args, **opts.mplayer_kwargs))
        flow.append("x, edge_index -> x")

        if opts.n_conv_layers >= 2:
            layers.append(get_layer(opts.activation_layer)())
            flow.append("x -> x")

            layers.append(get_layer("Dropout")(opts.dropout_ratio))
            flow.append("x -> x")

            hidden2 = int(opts.latent_dim*pow(opts.layer_ratio, opts.n_conv_layers-2))

            layers.append(get_mp_layer(opts.mplayer)(hidden1, hidden2, *opts.mplayer_args, **opts.mplayer_kwargs))
            flow.append("x, edge_index -> x")

        if opts.n_conv_layers == 3:
            layers.append(get_layer("Dropout")(opts.dropout_ratio))
            flow.append("x -> x")
            
            layers.append(get_mp_layer(opts.mplayer)(hidden2, opts.latent_dim, *opts.mplayer_args, **opts.mplayer_kwargs))
            flow.append("x, edge_index -> x")

        self.flow = flow
        self.nn = Sequential("x, edge_index", [(layer, flow) for layer, flow in zip(layers, flow)])

    def forward(self, x, edge_index):
        return self.nn(x, edge_index)
