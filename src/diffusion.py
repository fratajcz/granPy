import torch
from tqdm import tqdm
from src.utils import print_memory

class DiffusionWrapper(torch.nn.Module):
    def __init__(self, model, opts, device):
        super().__init__()
        self.model = model
        self.opts = opts
        self.eps = torch.tensor(1e-3, dtype=torch.float32)
        self.num_steps = opts.diffusion_steps
        self.noise = LogLinearNoise(self.eps)
        self.mode = 'train'
        self.device = device
        
        if opts.fixed_t is not None:
            if opts.fixed_t >= 0.0 and opts.fixed_t <= 1.0:
                self.t = torch.tensor(opts.fixed_t, dtype=torch.float32) 
            else:
                raise ValueError("Fixed time must be between 0 and 1")
        else:
            self.t = None
        
    def train_scores(self, x, e, neg_edges): ##check number of neg edges
            
        # forward diffusion process
        e0 = e
        t = self.sample_time() if self.t is None else self.t
        et, mask = self.mask_edges(e0, t)
        if self.opts.wandb_tracking: wandb.log({"t_sampled": t}, commit=False)
        
        # denoising prediction
        zt = self.model.encode(x, et)
        pos_out = self.model.decode(zt, e[mask], sigmoid=True)
        neg_out = self.model.decode(zt, neg_edges[mask], sigmoid=True)
        
        return pos_out, neg_out
    
    @torch.no_grad
    def eval_scores(self, x, target, pos_eval, neg_eval, unmask_topk=False, binarize=True):
        topk = pos_eval.shape[1] if unmask_topk else None
        e0_theta = self.sample(x, self.num_steps, target, num_edges=topk)
        
        if binarize:
            pos_out = e0_theta[pos_eval[0], pos_eval[1]].float()
            neg_out = e0_theta[neg_eval[0], neg_eval[1]].float()
        else:
            z = self.model.encode(x, e0_theta)
            pos_out = self.model.decode(z, pos_eval, sigmoid=True)
            neg_out = self.model.decode(z, neg_eval, sigmoid=True)
        return pos_out, neg_out

       
    @torch.no_grad
    def sample(self, x, num_steps, target, num_edges=None):
        self.dt = (1 - self.eps) / num_steps
        step_edges = None if num_edges is None else int(num_edges / num_steps)
        
        mask_t = torch.ones(x.shape[0], x.shape[0]).to(self.device)
        if target is not None:
            et = target
            mask_t[tuple(et)] = 0
        else:
            et = torch.empty(2, 0).to(self.device)
            
        for i in tqdm(range(num_steps)):
            if self.opts.verbose:
                print(f"diffusion step {i}: {print_memory(self.device)}")
            t = 1 - i * self.dt.to(self.device)
            
            zt = self.model.encode(x, et)
            adj0_theta = self.model.decoder.forward_all(zt, sigmoid=True)
            
            es_unmasked, mask_s = self.unmask_edges(adj0_theta, mask_t, t, step_edges)
            es = torch.hstack((et, es_unmasked))
            
            et = es
            mask_t.copy_(mask_s)
            
        return es
        
    def sample_time(self):
        pt = torch.rand(1).to(self.device)
        t = (1 - self.eps) * pt + self.eps
        return t

    def mask_edges(self, e0, t):
        ## TODO: add option to also switch negatives to positives or adj = 0.5 for mask
        self.sigma, self.dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-self.sigma)
        mask = (torch.rand(e0.shape[1]).to(self.device) < move_chance)
        et = e0[:,~mask]
        return et, mask

    @torch.no_grad
    def unmask_edges(self, adj0_theta, mask_t, t, top_edges = None):
        s = t - self.dt
        sigma_s, _ = self.noise(s)
        move_chance_s = 1 - torch.exp(-sigma_s)
        
        ## TODO: add option to only decode unmasked edges instead of all
        #move_chance_t = 1 - torch.exp(-sigma_t)
        #sigma_t, _ = self.noise(t)
        #n_unmask = int(move_chance_t - move_chance_s)
        
        ## Optimization: low rank bernoulli?
        
        if top_edges is None:
            adj0_sample = torch.bernoulli(adj0_theta).fill_diagonal_(0) 
        else:
            adj0_sample = torch.zeros_like(adj0_theta)
            _, top_indices = adj0_theta.fill_diagonal_(0).flatten().topk(top_edges)
            adj0_sample.view(-1)[top_indices] = 1
        mask_s= torch.bernoulli(torch.full_like(adj0_sample, move_chance_s)) * mask_t
        es_unmasked = adj0_sample * (1-mask_s) * mask_t
        es_unmasked = es_unmasked.nonzero().t()
        return es_unmasked, mask_s
        
    def diff_loss(self, e0_theta, e0):
        ## TODO: add batch
        log_p_theta = torch.nn.functional.binary_cross_entropy(e0_theta, e0, reduction='mean')
        loss = log_p_theta * (self.dsigma / torch.expm1(self.sigma))

        return loss
    
class LogLinearNoise(torch.nn.Module): #simplify to t?
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