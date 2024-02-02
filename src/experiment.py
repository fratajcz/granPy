from src.datasets import DatasetBootstrapper
from src.nn.models import GAE_Kipf
from src.utils import get_hash
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.utils import to_undirected
import os

class Experiment:
    def __init__(self, opts):
        self.opts = opts

        # TODO: get model hash and dataset hash? To save processing same dataset for different models?
        self.hash = get_hash(opts)

        self.dataset = DatasetBootstrapper(opts, hash=self.hash).get_dataset()

        self.model = GAE_Kipf(input_dim=self.dataset.train_data.x.shape[1], opts=opts)

        self.loss_function = BCEWithLogitsLoss()

        self.optimizer = Adam(self.model.parameters(), lr=opts.lr)

        self.lrscheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer)

        self.lowest_loss = 1e10

        self.initial_patience = opts.es_patience

        self.patience = opts.es_patience

        self.test_performance = None

    def run(self):
        try:
            self.load_model()
        except FileNotFoundError:
            for epoch in range(self.opts.epochs):
                self.train_step()

                did_improve = self.eval_step(target="val")

                if did_improve:
                    self.patience = self.initial_patience
                else:
                    self.patience -= 1

                if self.patience == 0:
                    break

            self.load_model()

        self.eval_step(target="test")

        return self.test_performance

    def train_step(self):
        self.model.train()
        self.model.zero_grad()

        data = self.dataset.train_data

        loss = self.get_loss(data)

        loss.backward()

        self.optimizer.step()

    def eval_step(self, target):
        self.model.eval()
        self.model.zero_grad()
        did_improve = False

        data = getattr(self.dataset, "{}_data".format(target))

        loss = self.get_loss(data)

        if target == "val":
            if loss < self.lowest_loss:
                self.lowest_loss = loss
                self.save_model()
                did_improve = True
            self.lrscheduler.step(loss)
        elif target == "test":
            self.test_performance = loss
            did_improve = None
        return did_improve
    
    def get_loss(self, data):
        pos_out = self.model(data.x, data.edge_index)

        neg_out = self.model(data.x, to_undirected(data.edge_label_index[:, data.edge_label == 0]))

        pos_loss = self.loss_function(pos_out, torch.ones_like(pos_out))
        neg_loss = self.loss_function(neg_out, torch.zeros_like(pos_out))

        return pos_loss + neg_loss

    def save_model(self):
        state_dict = {
            'model_state_dict': self.model.state_dict(),
        }
        if not os.path.exists(os.path.dirname(self.opts.model_path)):
            os.makedirs(os.path.dirname(self.opts.model_path))
        torch.save(state_dict, os.path.join(self.opts.model_path, self.hash + ".pt"))

    def load_model(self, path=None):
        '''loads a state dict either from the current run or from a given path'''
        path = os.path.join(self.opts.model_path, self.hash + ".pt")
        if torch.cuda.is_available():
            try:
                state_dict = torch.load(path)
            except RuntimeError:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
        else:
            # in case there is no cuda available, always load to cpu
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict["model_state_dict"])