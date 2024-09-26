import torch
from tqdm import tqdm

class DiffusionWrapper(torch.nn.Module):
    def __init__(self, model, opts, device):
        super().__init__()
        self.model = model
        self.opts = opts
        self.eps = 1e-16
        self.num_steps = opts.diffusion_steps
        self.noise = LogLinearNoise(self.eps)
        self.mode = 'train'
        self.device = device
        
    def encode(self, x, e): 
        self.num_nodes = x.shape[0]
        if self.mode == 'train':
            
            # forward diffusion process
            e0 = e
            t = self.sample_time()
            et, self.mask = self.mask_edges(e0, t)
            
            # encode masked graph
            zt = self.model.encode(x, et)
            return zt
        
        elif self.mode == 'eval':
            et = e
            e0_theta = self.sample(x, target=et)
            return e0_theta
    
    def decode(self, zt, eval_edges, *args, **kwargs):
        if self.mode == 'train':
        
            # denoising prediction
            e0_theta = self.model.decode(zt, eval_edges, sigmoid=True)
            return e0_theta[self.mask]
        
        elif self.mode == 'eval':
            e0_theta = zt

            result = torch.zeros(eval_edges.shape[1], dtype=torch.float32).to(self.device)
            for i in range(eval_edges.shape[1]):
                if ((e0_theta == eval_edges[:, i].unsqueeze(1)).all(dim=0)).any():
                    result[i] = 1
            return result
       
    @torch.no_grad() 
    def sample(self, x, target=None):
        self.dt = (1 - self.eps) / self.num_steps
        
        mask_t = torch.ones(x.shape[0], x.shape[0]).to(self.device)
        if target is not None:
            et = target
            mask_t[tuple(et)] = 0
        else:
            et = torch.empty(2, 0).to(self.device)
            
        for i in tqdm(range(self.num_steps)):
            t = torch.tensor(1 - i * self.dt).to(self.device)
            
            zt = self.model.encode(x, et)
            adj0_theta = self.model.decoder.forward_all(zt, sigmoid=True)
            
            es_unmasked, mask_s = self.unmask_edges(adj0_theta, mask_t, t)
            es = torch.hstack((et, es_unmasked))
            
            et = es
            mask_t = mask_s
            
        return es
        
    def sample_time(self):
        pt = torch.rand(1).to(self.device)
        t = (1 - self.eps) * pt + self.eps
        return t

    def mask_edges(self, e0, t):
        # todo: add option to also switch negatives to positives
        self.sigma, self.dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-self.sigma)
        mask = (torch.rand(e0.shape[1]).to(self.device) < move_chance)
        xt = e0[:,~mask]
        return xt, mask

    @torch.no_grad()
    def unmask_edges(self, adj0_theta, mask_t, t):
        s = t - self.dt
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(s)
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        n_unmask = int(move_chance_t - move_chance_s)
        
        adj0_sample = torch.bernoulli(adj0_theta).fill_diagonal_(0)
        mask_s= torch.bernoulli(torch.full_like(adj0_sample, move_chance_s)) * mask_t
        es_unmasked = adj0_sample * (1-mask_s) * mask_t
        es_unmasked = es_unmasked.nonzero().t()
        return es_unmasked, mask_s
        
    def diff_loss(self, e0_theta, e0):
        # double check continous formulation: no need to binarize time
        log_p_theta = torch.nn.functional.binary_cross_entropy(e0_theta, e0, reduction='none')
        loss = log_p_theta * (self.dsigma / torch.expm1(self.sigma))

        return loss.sum()

class LogLinearNoise(torch.nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        
    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)
        
    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def forward(self, t):
        # Assume time goes from 0 to 1
        return self.total_noise(t), self.rate_noise(t)