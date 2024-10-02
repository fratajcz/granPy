import unittest
from src.diffusion import DiffusionWrapper, LogLinearNoise
import src.nn.models as models
from src.utils import opts, get_dataset_hash
from src.datasets import DatasetBootstrapper
from src.negative_sampling import neg_sampling
import torch


class DiffusionTest(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        parameters = dict(
            # Data parameters
            dataset= "jackson",
            
            # Model parameters
            mplayer= "",
            mplayer_args= [],
            mplayer_kwargs= dict(),
            decoder= "InnerProductDecoder",
            model= "NaiveModel",
            encoder= "",
            layer_ratio = 0,
            n_conv_layers= 0,
            latent_dim = 0,
            dropout_ratio = 0,
            
            # Training/Evaluation parameters
            es_patience= 0,
            val_mode= "max",
            val_metric= "average_precision_score",
            test_metrics= ["average_precision_score", "roc_auc_score"],
            epochs= 1,
            negative_sampling= "random",
            score_batched= False,
            
            #General settings
            cuda= False,
            cache_model = False,
            wandb_tracking = False,
            wandb_save_model = False,
        )
        self._opts = opts(parameters)
        self.device = "cpu"
        dataset_hash = get_dataset_hash(self._opts)
        self.dataset = DatasetBootstrapper(self._opts, hash=dataset_hash).get_dataset()
        self.model = getattr(models, self._opts.model)(input_dim=self.dataset.train_data.x.shape[1], opts=self._opts).to(self.device)

    def test_init(self):
                
        diff_model = DiffusionWrapper(model=self.model,opts=self._opts, device=self.device)
        self.assertFalse(diff_model is None)

    def test_encode(self):
        
        diff_model = DiffusionWrapper(model=self.model,opts=self._opts, device=self.device)
        diff_model.mode = "train"
        z_diff = diff_model.encode(self.dataset.train_data.x, self.dataset.train_data.edge_index)
        z_model = self.model.encode(self.dataset.train_data.x, self.dataset.train_data.edge_index[:, diff_model.mask])
        self.assertTrue((z_diff == z_model).all())
        
    def test_decode(self):
        neg_samples = neg_sampling(self.dataset.train_data, space="full", type="random")

        diff_model = DiffusionWrapper(model=self.model,opts=self._opts, device=self.device)
        diff_model.mode = "train"
        zt = diff_model.encode(self.dataset.train_data.x, self.dataset.train_data.edge_index)
        pos_edges = diff_model.decode(zt, self.dataset.train_data.pos_edges)
        neg_edges = diff_model.decode(zt, neg_samples)
        
        model_pos_edges = self.model.decoder.forward(zt, self.dataset.train_data.pos_edges, sigmoid=True)[diff_model.mask]
        model_neg_edges = self.model.decoder.forward(zt, neg_samples, sigmoid=True)[diff_model.mask]
        self.assertTrue((pos_edges == model_pos_edges).all())
        self.assertTrue((neg_edges == model_neg_edges).all())
        
    def test_sample_empty(self):
        diff_model = DiffusionWrapper(model=self.model,opts=self._opts, device=self.device)
        e0_theta = diff_model.sample(self.dataset.train_data.x, target=None, num_steps = 1)
        
        pred_adj = self.model.decoder.forward_all(self.model.encode(self.dataset.train_data.x, self.dataset.train_data.edge_index), sigmoid=True)
        num_nodes = self.dataset.train_data.num_nodes
        e0_edge_count = torch.tensor([e0_theta.shape[1] + num_nodes], dtype=torch.float32)
        
        self.assertTrue(torch.isclose(e0_edge_count, pred_adj.sum(), rtol=1e-3))
        self.assertTrue((e0_theta[0] != e0_theta[1]).all())
    
    def test_sample_target(self):
        diff_model = DiffusionWrapper(model=self.model,opts=self._opts, device=self.device)
        e0_theta = diff_model.sample(self.dataset.train_data.x, target=self.dataset.train_data.edge_index, num_steps = 1)
        
        self.assertTrue(e0_theta.shape[1] > self.dataset.train_data.edge_index.shape[1])
        self.assertTrue(e0_theta.shape[1] < self.dataset.train_data.num_nodes**2)
        self.assertTrue((e0_theta[0] != e0_theta[1]).all())
        
        train_edges = self.dataset.train_data.edge_index
        result = torch.zeros(train_edges.shape[1], dtype=torch.int)
        for i in range(train_edges.shape[1]):
            if ((train_edges == e0_theta[:, i].unsqueeze(1)).all(dim=0)).any():
                result[i] = 1
        self.assertTrue(result.all())
        
    def test_sample_empty_topk(self):
        
        num_edges=20
        diff_model = DiffusionWrapper(model=self.model,opts=self._opts, device=self.device)
        e0_theta = diff_model.sample(self.dataset.train_data.x, target=None, num_steps = 1, num_edges=num_edges)
        
        self.assertTrue(e0_theta.shape[1] == num_edges)
        self.assertTrue((e0_theta[0] != e0_theta[1]).all())
        
    def test_sample_target_topk(self):
        
        num_edges=30
        diff_model = DiffusionWrapper(model=self.model,opts=self._opts, device=self.device)
        e0_theta = diff_model.sample(self.dataset.train_data.x, target=self.dataset.train_data.edge_index, num_steps = 1, num_edges=num_edges)
        
        self.assertTrue(e0_theta.shape[1] == num_edges + self.dataset.train_data.edge_index.shape[1])
        self.assertTrue((e0_theta[0] != e0_theta[1]).all())
        
    def test_sample_time(self):
        diff_model = DiffusionWrapper(model=self.model,opts=self._opts, device=self.device)
        max_time = 0
        min_time = 1
        for i in range(1000):
            t = diff_model.sample_time()
            max_time = max(max_time, t)
            min_time = min(min_time, t)
            
        self.assertTrue(max_time < 1)
        self.assertTrue(min_time > 0)
        
    def test_diff_loss(self):
        test_edges = torch.tensor([1, 0], dtype=torch.float32)
        pred_A = torch.tensor([1, 0], dtype=torch.float32)
        pred_B = torch.tensor([0.5, 0.5], dtype=torch.float32)
        pred_C = torch.tensor([0, 1], dtype=torch.float32)
        
        diff_model = DiffusionWrapper(model=self.model,opts=self._opts, device=self.device)
        diff_model.dsigma = torch.tensor(0.01, dtype=torch.float32)
        diff_model.sigma = torch.tensor(0.5, dtype=torch.float32)
        loss_A = diff_model.diff_loss(pred_A, test_edges)
        loss_B = diff_model.diff_loss(pred_B, test_edges)
        loss_C = diff_model.diff_loss(pred_C, test_edges)
        
        self.assertTrue(loss_A < loss_B)
        self.assertTrue(loss_B < loss_C)
        
        
    def test_mask_edges(self):
        edges = torch.tensor([[0, 1, 2, 3],
                              [1, 2, 3, 0]])
        diff_model = DiffusionWrapper(model=self.model,opts=self._opts, device=self.device)
        e0, mask0 = diff_model.mask_edges(edges, torch.tensor([0]))
        e1, mask1 = diff_model.mask_edges(edges, torch.tensor([1]))
        e_5, mask_5 = diff_model.mask_edges(edges, torch.tensor([0.5]))
        
        self.assertFalse(mask0.any())
        self.assertTrue(mask1.all())
        self.assertTrue((e0 == edges).all())
        self.assertTrue(e1.shape[1] == 0)
        self.assertTrue((e_5 == edges[:, ~mask_5]).all())
        self.assertTrue(e_5.shape[1] < edges.shape[1])
        
    def test_unmask_edges(self):
        diff_model = DiffusionWrapper(model=self.model,opts=self._opts, device=self.device)
        t = 0.5
        diff_model.dt = 1-t
        adj0_theta = torch.tensor([[1, 1, 0],
                                   [0, 1, 0],
                                   [1, 0, 1]], dtype=torch.float32)
        mask_t = torch.tensor([[0, 0, 0],
                                [1, 0, 0],
                                [1, 1, 0]], dtype=torch.float32)
        e_unmasked, e_mask = diff_model.unmask_edges(adj0_theta, mask_t, t)
        self.assertTrue((e_unmasked == torch.tensor([[2],[0]], dtype=torch.float32)).all())
        self.assertTrue((e_mask == torch.tensor([[0, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 0]], dtype=torch.float32)).all())
        
    def test_loglinearnoise(self):
        noise = LogLinearNoise(eps=torch.tensor(0.001))
        one = torch.ones(1, dtype=torch.float32)
        zero = torch.zeros(1, dtype=torch.float32)
        
        self.assertTrue((1 - torch.exp(-noise(zero)[0])) == 0)
        self.assertTrue(torch.isclose(1 - torch.exp(-noise(one)[0]), one, rtol=1e-3))
        self.assertTrue((1 - torch.exp(-noise(one)[0])) < one)
        
