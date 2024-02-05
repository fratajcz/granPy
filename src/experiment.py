from src.datasets import DatasetBootstrapper
from src.nn.models import GAE_Kipf
from src.utils import get_hash
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.utils import to_undirected
import os
import sklearn.metrics as skmetrics
import copy
from tqdm import tqdm

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

        if self.opts.val_mode == "max":
            self.best_val_performance = -1e10
        elif self.opts.val_mode == "min":
            self.best_val_performance = 1e10
        else:
            raise ValueError("Validation Mode can either be max or min, but not {}".format(self.opts.val_mode))

        self.initial_patience = opts.es_patience

        self.patience = opts.es_patience

        self.test_performance = None

    def run(self):
        try:
            self.load_model()
        except FileNotFoundError:
            for epoch in (pbar := tqdm(range(self.opts.epochs))):
                pbar.set_description("Best {}: {}".format(self.opts.val_metric, self.best_val_performance))
                self.train_step()

                did_improve = self.eval_step(target="val")

                if self.out_of_patience(did_improve):
                    break

            self.load_model()

        self.eval_step(target="test")

        return self.test_performance

    def train_step(self):
        self.model.train()
        self.model.zero_grad()

        data = self.dataset.train_data

        loss, _, _ = self.get_loss(data)

        loss.backward()

        self.optimizer.step()

    def eval_step(self, target):
        self.model.eval()
        self.model.zero_grad()
        did_improve = False

        data = getattr(self.dataset, "{}_data".format(target))

        loss, pos_out, neg_out = self.get_loss(data)

        if target == "val":
            value = self.get_metric(pos_out, neg_out, [self.opts.val_metric])[self.opts.val_metric]
            if self.is_best_val_performance(value):
                self.best_val_performance = value
                self.save_model()
                did_improve = True
            self.lrscheduler.step(loss)
        elif target == "test":
            self.test_performance = self.get_metric(pos_out, neg_out, self.opts.test_metrics)
            did_improve = None
        return did_improve
    
    def is_best_val_performance(self, value):
        if self.opts.val_mode == "max":
            return self.best_val_performance < value
        else:
            return self.best_val_performance > value
    
    def get_loss(self, data):
        z = self.model.encode(data.x, data.edge_index)

        pos_out = self.model.decode(z, data.edge_index)

        # TODO adapt this step to allow scoring against pot_net
        neg_out = self.model.decode(z, to_undirected(data.edge_label_index[:, data.edge_label == 0]))

        pos_loss = self.loss_function(pos_out, torch.ones_like(pos_out))
        neg_loss = self.loss_function(neg_out, torch.zeros_like(neg_out))

        return pos_loss + neg_loss, pos_out, neg_out

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

    def out_of_patience(self, did_improve: bool):
        if did_improve:
            self.patience = self.initial_patience
        else:
            self.patience -= 1

        return self.patience == 0

    def get_metric(self, pos_out: torch.Tensor, neg_out: torch.Tensor, metrics: list):
        all_out = torch.hstack((pos_out.squeeze(), neg_out.squeeze()))
        all_labels = torch.hstack((torch.ones_like(pos_out.squeeze()), torch.zeros_like(neg_out.squeeze())))

        values = {}
        for metric in metrics:
            function = getattr(skmetrics, metric)
            values.update({metric: function(y_true=all_labels.detach().cpu().numpy(), y_score=all_out.detach().cpu().numpy())})

        return values
    

class CVWrapper:
    def __init__(self, opts):
        self.opts = opts

        self.val_seeds = range(opts.val_seed, opts.val_seed + opts.n_folds)
        self.performance_reports = None

    def run(self):
        self.performance_reports = []
        for val_seed in tqdm(self.val_seeds):
            subopts = copy.deepcopy(self.opts)
            subopts.val_seed = val_seed
            fold_experiment = Experiment(subopts)
            self.performance_reports.append(fold_experiment.run())

        return self.performance_reports
        