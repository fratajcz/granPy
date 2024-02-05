from src.nn.encoders import GAE_Encoder
import src.nn.decoders as decoders
from torch_geometric.nn.models import GAE


class GAE_Kipf(GAE):
    def __init__(self, input_dim, opts):
        encoder = GAE_Encoder(input_dim, opts)
        decoder = getattr(decoders, opts.decoder)(opts)
        super(GAE_Kipf, self).__init__(encoder=encoder, decoder=decoder)