import torch
from tqdm import tqdm

class DiffusionWrapper(torch.nn.Module):
    def __init__(self, model, opts, device):
        super().__init__()
        self.model = model
        self.opts = opts
        self.eps = torch.tensor(1e-16, dtype=torch.float32)
        self.num_steps = opts.diffusion_steps
        self.noise = LogLinearNoise(self.eps)
        self.mode = 'train'
        self.device = device
        
    def encode(self, x, e): 
        self.num_nodes = x.shape[0]
            
        # forward diffusion process
        e0 = e
        t = self.sample_time()
        et, self.mask = self.mask_edges(e0, t)
        
        # encode masked graph
        zt = self.model.encode(x, et)
        return zt
    
    def decode(self, zt, eval_edges, *args, **kwargs):

        # denoising prediction
        e0_theta = self.model.decode(zt, eval_edges, sigmoid=True)
        return e0_theta[self.mask]
    
    @torch.no_grad
    def eval_edges(self, x, target, pos_eval, neg_eval):
        e0_theta = self.sample(x, target=target, num_steps=self.num_steps)
        
        pos_out = torch.zeros(pos_eval.shape[1], dtype=torch.float32).to(self.device)
        for i in range(pos_eval.shape[1]):
            if ((e0_theta == pos_eval[:, i].unsqueeze(1)).all(dim=0)).any():
                pos_out[i] = 1
        neg_out = torch.zeros(neg_eval.shape[1], dtype=torch.float32).to(self.device)
        for i in range(neg_eval.shape[1]):
            if ((e0_theta == neg_eval[:, i].unsqueeze(1)).all(dim=0)).any():
                neg_out[i] = 1
        return pos_out, neg_out

       
    @torch.no_grad
    def sample(self, x, target=None, num_steps=None):
        self.dt = (1 - self.eps) / num_steps
        
        mask_t = torch.ones(x.shape[0], x.shape[0]).to(self.device)
        if target is not None:
            et = target
            mask_t[tuple(et)] = 0
        else:
            et = torch.empty(2, 0).to(self.device)
            
        for i in tqdm(range(num_steps)):
            t = 1 - i * self.dt.to(self.device)
            
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
        et = e0[:,~mask]
        return et, mask

    @torch.no_grad
    def unmask_edges(self, adj0_theta, mask_t, t):
        s = t - self.dt
        #sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(s)
        #move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        #n_unmask = int(move_chance_t - move_chance_s)
        
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