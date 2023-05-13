import numpy as np
import torch
from torch import nn

import utils as utils
from utils.helpers import cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule, \
                            extract, Losses

class GaussianDiffusion(nn.Module):
    def __init__(self, model, observation_dim, action_dim, cond_dim, horizon,
                 n_timesteps=100, loss_type='l2', clip_denoised=True, predict_epsilon=True,
                 loss_discount=1.0, loss_weights=None, conditional=False, action_weight=1.,
                 condition_guidance_w=0.1, beta_schedule='cosine', device='cpu'):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.obsact_dim = observation_dim + action_dim
        self.transition_dim = (observation_dim*2 + action_dim + 2) * horizon - observation_dim
        self.cond_dim = cond_dim
        
        self.model = model
        self.conditional = conditional
        self.condition_guidance_w = condition_guidance_w
        self.device = device
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'linear':
            betas = vp_beta_schedule(n_timesteps)
        else:
            NotImplementedError(beta_schedule)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.to(device))
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to(device))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod).to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod -1).to(device))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance.to(device))
        
        ## log calculation clipped because the posterior variance 
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)).to(device))
        self.register_buffer('posterior_mean_coef1', (betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(device))
        self.register_buffer('posterior_mean_coef2', ((1. -  alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(device))
        
        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory
            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)
        
        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w
        
        ## decay loss with trajectory timestep: discount**t
        # discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        # discounts = discounts / discounts.mean()
        # loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        loss_weights = dim_weights * discount
        
        ## manually set a0 weight
        loss_weights[:self.action_dim] = action_weight
        return loss_weights.to(self.device)
    
    @torch.no_grad()
    def score(self, x, t, state, cond):
        if self.conditional:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, t, state, cond, use_dropout=False)
            epsilon_uncond = self.model(x, t, state, cond, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, t, state, cond)        
        return - extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)**(-1) * epsilon
    
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            q(x_t | x_0) --> x_0 = a * x_t - b * epsilon
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
        
    def q_posterior(self, x_start, x_t, t):
        '''
            q(x_{t-1} | x_t, x_0) ~ N( mu_tilde(x_t, x_0), beta_tilde )
        '''
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, state, cond):
        '''
            p_theta(x_{t-1} | x_t)
        '''
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
        
        if self.conditional:
            assert cond != None
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, t, state, cond, use_dropout=False)
            epsilon_uncond = self.model(x, t, state, cond, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            # assert cond == None
            epsilon = self.model(x, t, state, cond)
                
        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
   
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, state, cond):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, state=state, cond=cond)
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t==0).float()).reshape(b, *((1,) * (len(x.shape)-1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, state, cond, return_diffusion=False):
        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=self.device)
        
        if return_diffusion: diffusion = [x]
        
        progress = utils.Progress(self.n_timesteps)
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state, cond)
            
            progress.update({'t': i})
            if return_diffusion: diffusion.append(x)
        progress.close()
        
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        
    @torch.no_grad()
    def conditional_sample(self, state, cond, *args, **kwargs):
        batch_size = cond.shape[0]
        shape =  (batch_size, self.transition_dim)
        return self.p_sample_loop(shape, state, cond, *args, **kwargs)
    
    #------------------------------------------ training ------------------------------------------#
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample
    
    def p_losses(self, x_start, t, state, cond):
        noise = torch.randn_like(x_start)
        b, *_ = x_start.shape
        x_start = x_start.float()
        
        x_noisy = self.q_sample(x_start, t, noise)
        
        if self.model.calc_energy:
            assert self.predict_epsilon
            x_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            cond.requires_grad = True
            noise.requires_grad = True
        
        x_recon = self.model(x_noisy, t, state, cond)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        
        if self.predict_epsilon:
            pred_xstart = self.predict_start_from_noise(x_noisy, t, x_recon)
        else:
            pred_xstart = x_recon
        
        assert noise.shape == x_recon.shape
        
        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise)
        else:
            loss = self.loss_fn(x_recon, x_start)
            
        return loss
    
    def loss(self, trajectories, cond):
        batch_size = len(trajectories)
        x = trajectories[..., self.observation_dim:]
        state = trajectories[..., :self.observation_dim]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        loss = self.p_losses(x, t, state, cond)
    
        return loss.mean()

    def forward(self, state, cond, *args, **kwargs):
        return self.conditional_sample(state, cond, *args, **kwargs)