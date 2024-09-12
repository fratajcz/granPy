from src.nn.encoders import GAE_Encoder, IdentityEncoder
import src.nn.decoders as decoders
from torch_geometric.nn.models import GAE
import torch


class AutoEncoder(GAE):
    def __init__(self, input_dim, opts):
        encoder = GAE_Encoder(input_dim, opts)
        decoder = getattr(decoders, opts.decoder)(opts)
        super(AutoEncoder, self).__init__(encoder=encoder, decoder=decoder)

class NaiveModel(torch.nn.Module):
    def __init__(self, input_dim, opts):
        super(NaiveModel, self).__init__()
        self.dummy_parameter = (torch.nn.Parameter(torch.rand((1,))))
        self.decoder = getattr(decoders, opts.decoder)(opts)
    
    def encode(self, x, *args, **kwargs):
        return x
    
    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)