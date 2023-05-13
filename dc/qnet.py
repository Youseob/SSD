import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from utils.helpers import soft_copy_nn_module, minuscosine
from utils.arrays import to_np, to_torch

class CQLCritic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, normalizer, hidden_dim=256, gamma=0.95, 
                 min_q_weight=1.0, temp=1.0, n_random=10, max_q_backup=False):
        super(CQLCritic, self).__init__()
        self.qf1 = nn.Sequential(nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.qf2 = nn.Sequential(nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        
        self.gamma = gamma
        self.n_random = n_random
        self.min_q_weight = min_q_weight
        self.temp = temp
        self.max_q_backup = max_q_backup

        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.obsact_dim = state_dim + action_dim
        
        self.normalizer = normalizer
        if 'goals' in self.normalizer.normalizers:
            self.goal_key = 'goals'
        elif 'rtgs' in self.normalizer.normalizers:
            self.goal_key = 'rtgs'
        
    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1(x), self.qf2(x)
    
    def forward_target(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1_target(x), self.qf2_target(x)
    
    def q1(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1(x)

    def q_min(self, state, action, goal):
        q1, q2 = self.forward(state, action, goal)
        return torch.min(q1, q2)
    
    def target_update(self):
        soft_copy_nn_module(self.qf1, self.qf1_target)
        soft_copy_nn_module(self.qf2, self.qf2_target)
    
    def loss(self, trajectories, goal, ema_model):
        batch_size = trajectories.shape[0]
        s = trajectories[:, :self.observation_dim]
        a, ns, r, done = self._normalized_transition(trajectories[:, self.observation_dim:])
        a_new, ns_new, r_new, done_new = self._normalized_transition(ema_model(s, goal))
        r = self.unnorm(r, 'rewards')
        r_new = self.unnorm(r_new, 'rewards')
        
        pred_q1, pred_q2 = self.forward(self.unnorm(s, 'observations'), self.unnorm(a, 'actions'), self.unnorm(goal, self.goal_key))
        pred_q = torch.min(pred_q1, pred_q2)
        if self.max_q_backup:
            # data
            ns_rpt = torch.repeat_interleave(ns, repeats=10, dim=0)
            samples = ema_model(ns_rpt, goal)
            na_rpt, *_ = self._normalized_transition(samples)
            pred_targ_q1, pred_targ_q2 = self.forward_target(
                self.unnorm(ns_rpt, 'observations'), self.unnorm(na_rpt, 'actions'), self.unnorm(goal, self.goal_key)
                )
            targ_q1 = pred_targ_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            targ_q2 = pred_targ_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            pred_targ_q = torch.min(targ_q1, targ_q2)
            
            # sample
            ns_rpt_new = torch.repeat_interleave(ns_new, repeats=10, dim=0)
            na_rpt_new , *_ = self._normalized_transition(ema_model(ns_rpt_new, goal))
            pred_targ_q1_new, pred_targ_q2_new = self.forward_target(
                self.unnorm(ns_rpt_new, 'observations'), self.unnorm(na_rpt_new, 'actions'), self.unnorm(goal, self.goal_key)
                )
            targ_q1_new = pred_targ_q1_new.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            targ_q2_new = pred_targ_q2_new.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            targ_q_new = torch.min(targ_q1_new, targ_q2_new)
        else:            
            # data
            samples = ema_model(ns, goal)
            na, *_ = self._normalized_transition(samples)
            pred_targ_q1, pred_targ_q2 = self.forward_target(
                self.unnorm(ns, 'observations'), self.unnorm(na, 'actions'), self.unnorm(goal, self.goal_key)
                )
            pred_targ_q = torch.min(pred_targ_q1, pred_targ_q2)

            # sample
            # ns_unrm_new = to_torch(self.normalizer.unnormalize(to_np(ns_new), 'observations'))
            na_new, *_ = self._normalized_transition(ema_model(ns_new, goal))
            curr_q1, curr_q2 = self.forward(self.unnorm(s, 'observations'), self.unnorm(a_new, 'actions'), self.unnorm(goal, self.goal_key))
            targ_q1, targ_q2 = self.forward_target(self.unnorm(ns_new, 'observations'), self.unnorm(na_new, 'actions'), self.unnorm(goal, self.goal_key))
            targ_q_new = torch.min(targ_q1, targ_q2)     
        
        r = r.reshape(list(r.shape) + [1])
        done = done.reshape(list(done.shape) + [1])
        pred_td_target = (pred_targ_q * self.gamma * (1.-done) + r).detach() 

        r_new = r_new.reshape(list(r_new.shape) + [1])
        done_new = done_new.reshape(list(done_new.shape) + [1])
        td_target = (targ_q_new * self.gamma * (1.-done_new) + r_new).detach()        
        
        q1_loss = 0.5*F.mse_loss(pred_q1, td_target, reduction='mean') + 0.5*F.mse_loss(pred_q1, pred_td_target, reduction='mean')
        q2_loss = 0.5*F.mse_loss(pred_q2, td_target, reduction='mean') + 0.5*F.mse_loss(pred_q2, pred_td_target, reduction='mean')
        
        # random
        random_actions = np.random.uniform(-1, 1, (curr_q2.shape[0], self.n_random, a_new.shape[-1]))
        random_actions = to_torch(random_actions).to('cuda')
        s_new_temp = einops.repeat(s, 'b d -> b n d', n=self.n_random)
        goal_temp = einops.repeat(goal, 'b d -> b n d', n=self.n_random)
        rand_q1, rand_q2 = self.forward(self.unnorm(s_new_temp, 'observations'), random_actions, self.unnorm(goal_temp, self.goal_key))
        
        # next
        next_q1, next_q2 = self.forward(self.unnorm(s, 'observations'), self.unnorm(na_new, 'actions'), self.unnorm(goal, self.goal_key))
        
        # aggregate
        cat_q1 = torch.cat(
            [rand_q1, pred_q1.unsqueeze(1), next_q1.unsqueeze(1), curr_q1.unsqueeze(1)], 1
        )
        cat_q2 = torch.cat(
            [rand_q2, pred_q2.unsqueeze(1), next_q2.unsqueeze(1), curr_q2.unsqueeze(1)], 1
        )
        # std_q1 = torch.std(cat_q1, dim=1)
        # std_q2 = torch.std(cat_q2, dim=1)
        
        min_q1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.min_q_weight * self.temp
        min_q2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.min_q_weight * self.temp
        
        min_q1_loss = min_q1_loss - pred_q1.mean() * self.min_q_weight 
        min_q2_loss = min_q2_loss - pred_q2.mean() * self.min_q_weight 
        
        q1_loss_cql = q1_loss + min_q1_loss
        q2_loss_cql = q2_loss + min_q2_loss
        
        # [0, 1]
        # q_normed = (curr_q - curr_q.min()) / (curr_q.max() - curr_q.min()).mean()
        
        return q1_loss_cql, q2_loss_cql, q1_loss, q2_loss, pred_q
    
    
    def _normalized_transition(self, trajectories):
        
        # samples are normed values
        a = trajectories[:, :self.action_dim]
        ns = trajectories[:, self.action_dim:self.obsact_dim]
        r = trajectories[:, -2]
        done = (trajectories[:, -1] > 0.5).float()
        
        return a, ns, r, done

    def _unnormalized_transition(self, trajectories):
        
        # samples are normed values
        a = trajectories[:, :self.action_dim]
        ns = trajectories[:, self.action_dim:self.obsact_dim]
        r = trajectories[:, -2]
        done = (trajectories[:, -1] > 0.5).float()
        
        a = to_torch(self.normalizer.unnormalize(to_np(a), 'actions'))
        ns = to_torch(self.normalizer.unnormalize(to_np(ns), 'observations'))
        r = to_torch(self.normalizer.unnormalize(to_np(r), 'rewards'))
        
        return a, ns, r, done
    
    def unnorm(self, x, key):
        return to_torch(self.normalizer.unnormalize(to_np(x), key))
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, normalizer, hidden_dim=256, gamma=0.95, 
                 min_q_weight=1.0, temp=1.0, n_random=10, max_q_backup=False):
        super(Critic, self).__init__()
        self.qf1 = nn.Sequential(nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.qf2 = nn.Sequential(nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        
        self.gamma = gamma
        self.n_random = n_random
        self.min_q_weight = min_q_weight
        self.temp = temp
        self.max_q_backup = max_q_backup

        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.obsact_dim = state_dim + action_dim
        
        self.normalizer = normalizer
        if 'goals' in self.normalizer.normalizers:
            self.goal_key = 'goals'
        elif 'rtgs' in self.normalizer.normalizers:
            self.goal_key = 'rtgs'
        
    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1(x), self.qf2(x)
    
    def forward_target(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1_target(x), self.qf2_target(x)
    
    def q1(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1(x)

    def q_min(self, state, action, goal):
        q1, q2 = self.forward(state, action, goal)
        return torch.min(q1, q2)
    
    def target_update(self):
        soft_copy_nn_module(self.qf1, self.qf1_target)
        soft_copy_nn_module(self.qf2, self.qf2_target)
    
    def loss(self, trajectories, goal, ema_model):
        batch_size = trajectories.shape[0]
        s = trajectories[:, :self.observation_dim]
        a, ns, r, done = self._normalized_transition(trajectories[:, self.observation_dim:])
        a_new, ns_new, r_new, done_new = self._normalized_transition(ema_model(s, goal))
        r = self.unnorm(r, 'rewards')
        r_new = self.unnorm(r_new, 'rewards')
        
        pred_q1, pred_q2 = self.forward(self.unnorm(s, 'observations'), self.unnorm(a, 'actions'), self.unnorm(goal, self.goal_key))
        pred_q = torch.min(pred_q1, pred_q2)
        if self.max_q_backup:
            # data
            ns_rpt = torch.repeat_interleave(ns, repeats=10, dim=0)
            samples = ema_model(ns_rpt, goal)
            na_rpt, *_ = self._normalized_transition(samples)
            pred_targ_q1, pred_targ_q2 = self.forward_target(
                self.unnorm(ns_rpt, 'observations'), self.unnorm(na_rpt, 'actions'), self.unnorm(goal, self.goal_key)
                )
            targ_q1 = pred_targ_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            targ_q2 = pred_targ_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            pred_targ_q = torch.min(targ_q1, targ_q2)
            
            # sample
            ns_rpt_new = torch.repeat_interleave(ns_new, repeats=10, dim=0)
            na_rpt_new , *_ = self._normalized_transition(ema_model(ns_rpt_new, goal))
            pred_targ_q1_new, pred_targ_q2_new = self.forward_target(
                self.unnorm(ns_rpt_new, 'observations'), self.unnorm(na_rpt_new, 'actions'), self.unnorm(goal, self.goal_key)
                )
            targ_q1_new = pred_targ_q1_new.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            targ_q2_new = pred_targ_q2_new.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            targ_q_new = torch.min(targ_q1_new, targ_q2_new)
        else:            
            # data
            samples = ema_model(ns, goal)
            na, *_ = self._normalized_transition(samples)
            pred_targ_q1, pred_targ_q2 = self.forward_target(
                self.unnorm(ns, 'observations'), self.unnorm(na, 'actions'), self.unnorm(goal, self.goal_key)
                )
            pred_targ_q = torch.min(pred_targ_q1, pred_targ_q2)

            # sample
            # ns_unrm_new = to_torch(self.normalizer.unnormalize(to_np(ns_new), 'observations'))
            na_new, *_ = self._normalized_transition(ema_model(ns_new, goal))
            # curr_q1, curr_q2 = self.forward(self.unnorm(s, 'observations'), self.unnorm(a_new, 'actions'), self.unnorm(goal, self.goal_key))
            targ_q1, targ_q2 = self.forward_target(self.unnorm(ns_new, 'observations'), self.unnorm(na_new, 'actions'), self.unnorm(goal, self.goal_key))
            targ_q_new = torch.min(targ_q1, targ_q2)     
        
        r = r.reshape(list(r.shape) + [1])
        done = done.reshape(list(done.shape) + [1])
        pred_td_target = (pred_targ_q * self.gamma * (1.-done) + r).detach() 

        r_new = r_new.reshape(list(r_new.shape) + [1])
        done_new = done_new.reshape(list(done_new.shape) + [1])
        td_target = (targ_q_new * self.gamma * (1.-done_new) + r_new).detach()        
        
        q1_loss = 0.5*F.mse_loss(pred_q1, td_target, reduction='mean') + 0.5*F.mse_loss(pred_q1, pred_td_target, reduction='mean')
        q2_loss = 0.5*F.mse_loss(pred_q2, td_target, reduction='mean') + 0.5*F.mse_loss(pred_q2, pred_td_target, reduction='mean')
      
        
        return q1_loss, q2_loss, pred_q
    
    
    def _normalized_transition(self, trajectories):
        
        # samples are normed values
        a = trajectories[:, :self.action_dim]
        ns = trajectories[:, self.action_dim:self.obsact_dim]
        r = trajectories[:, self.obsact_dim]
        done = (trajectories[:, self.obsact_dim+1] > 0.5).float()
        
        return a, ns, r, done

    def _unnormalized_transition(self, trajectories):
        
        # samples are normed values
        a = trajectories[:, :self.action_dim]
        ns = trajectories[:, self.action_dim:self.obsact_dim]
        r = trajectories[:, self.obsact_dim]
        done = (trajectories[:, self.obsact_dim+1] > 0.5).float()
        
        a = to_torch(self.normalizer.unnormalize(to_np(a), 'actions'))
        ns = to_torch(self.normalizer.unnormalize(to_np(ns), 'observations'))
        r = to_torch(self.normalizer.unnormalize(to_np(r), 'rewards'))
        
        return a, ns, r, done
    
    def unnorm(self, x, key):
        return to_torch(self.normalizer.unnormalize(to_np(x), key))
    
class HindsightCritic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, normalizer, hidden_dim=256, gamma=0.95, 
                 min_q_weight=1.0, temp=1.0, n_random=50, max_q_backup=False):
        super(HindsightCritic, self).__init__()
        self.qf1 = nn.Sequential(nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.qf2 = nn.Sequential(nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)

        self.gamma = gamma
        self.n_random = n_random
        self.min_q_weight = min_q_weight
        self.temp = temp
        self.max_q_backup = max_q_backup

        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.obsact_dim = state_dim + action_dim
        self.goal_dim = goal_dim
        
        self.normalizer = normalizer
        if 'goals' in self.normalizer.normalizers:
            self.goal_key = 'goals'
        elif 'rtgs' in self.normalizer.normalizers:
            self.goal_key = 'rtgs'
            
    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1(x), self.qf2(x)
    
    def forward_target(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1_target(x), self.qf2_target(x)
    
    def q1(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1(x)

    def q_min(self, state, action, goal):
        q1, q2 = self.forward(state, action, goal)
        return torch.min(q1, q2)
    
    def target_update(self):
        soft_copy_nn_module(self.qf1, self.qf1_target)
        soft_copy_nn_module(self.qf2, self.qf2_target)
    
    def loss(self, batch, ema_model):
        trajectories = batch.trajectories
        batch_size = len(trajectories)
        observation = trajectories[:, :self.observation_dim]
        action = trajectories[:, self.observation_dim:self.obsact_dim]
        next_observation = trajectories[:, self.obsact_dim:self.obsact_dim+self.observation_dim]
        reward = trajectories[:, self.obsact_dim+self.observation_dim].reshape((batch_size,1))
        samples = ema_model(next_observation, batch.conditions, batch.goals)
        action_pi = samples[:, :self.action_dim]
        
        td_target = torch.zeros((batch_size, 1), device=trajectories.device)
        at_goal = (next_observation[:, :self.goal_dim] == batch.goals).all(1)
        td_target[at_goal] = 1
        q1, q2 = self.forward_target(next_observation, action_pi, batch.goals)
        td_target[~at_goal] = reward.reshape((batch_size,1)) + self.gamma * torch.min(q1, q2)
        
        pred_q1, pred_q2 = self.forward(observation, action, batch.goals)
        pred_q = torch.min(pred_q1, pred_q2)
        
        random_actions = np.random.uniform(-1, 1, (pred_q.shape[0], self.n_random, action.shape[-1]))
        random_actions = self.unnorm(to_torch(random_actions).to('cuda'), 'actions')
        observation_temp = self.unnorm(einops.repeat(observation, 'b d -> b n d', n=self.n_random), 'observations')
        goal_temp = self.unnorm(einops.repeat(batch.goals, 'b d -> b n d', n=self.n_random), self.goal_key)
        rand_q1_rpt, rand_q2_rpt = self.forward_target(observation_temp, random_actions, goal_temp)
        rand_q1 = rand_q1_rpt.view(batch_size, self.n_random).max(dim=1, keepdim=True)[0]
        rand_q2 = rand_q2_rpt.view(batch_size, self.n_random).max(dim=1, keepdim=True)[0]
        min_q1_loss = torch.logsumexp(rand_q1 / self.temp, dim=1).mean() * self.min_q_weight * self.temp
        min_q2_loss = torch.logsumexp(rand_q2 / self.temp, dim=1).mean() * self.min_q_weight * self.temp
        
        loss1 = F.mse_loss(td_target.detach(), pred_q1, reduction='mean') + min_q1_loss
        loss2 = F.mse_loss(td_target.detach(), pred_q2, reduction='mean') + min_q2_loss
        
        return loss1, loss2, pred_q

    def unnorm(self, x, key):
        return to_torch(self.normalizer.unnormalize(to_np(x), key))