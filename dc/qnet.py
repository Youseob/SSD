import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.distributions import Exponential

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
    def __init__(self, state_dim, action_dim, goal_dim, dataset, hidden_dim=256, gamma=0.99, 
                 min_q_weight=1.0, temp=1.0, n_random=50, max_q_backup=False):
        super(HindsightCritic, self).__init__()
        self.qf1 = nn.Sequential(nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1),
                                      )

        self.qf2 = nn.Sequential(nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1),
                                      )
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        self.actor = nn.Sequential(nn.Linear(state_dim+goal_dim, hidden_dim),
                                   nn.Mish(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.Mish(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.Mish(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.Mish(),
                                   nn.Linear(hidden_dim, action_dim),
                                   )

        self.gamma = gamma
        self.n_random = n_random
        self.min_q_weight = min_q_weight
        self.temp = temp
        self.max_q_backup = max_q_backup

        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.obsact_dim = state_dim + action_dim
        self.goal_dim = goal_dim
        
        try:
            self.has_object = dataset.env.has_object
        except:
            self.has_object = False
            
        self.env_name = dataset.env.name
        self.normalizer = dataset.normalizer
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
    
    def forward_actor(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return torch.clamp(self.actor(x), -1, 1)
    
    def q1(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1(x)

    def q_min(self, state, action, goal):
        q1, q2 = self.forward(state, action, goal)
        return torch.min(q1, q2)
    
    def target_update(self):
        soft_copy_nn_module(self.qf1, self.qf1_target)
        soft_copy_nn_module(self.qf2, self.qf2_target)
    
    def loss(self, batch, goal_rand, ema_model):
        trajectories = batch.trajectories
        batch_size, horizon, _ = trajectories.shape
        
        # hindsight goals and values
        if self.has_object:
            # Hindsight goals
            ag = self.unnorm(batch.trajectories[:, :, self.goal_dim:2*self.goal_dim], 'achieved_goals')
            dg = self.unnorm(goal_rand, 'goals')
            observation = self.unnorm(trajectories[:, 0, :self.observation_dim], 'observations')
            action = self.unnorm(trajectories[:, 0, self.observation_dim:], 'actions')
            observation_cat = observation.repeat(2,1)
            action_cat = action.repeat(2,1)
            future_t = np.random.randint(1, horizon, size=(batch_size,))
            hindsight_goals = ag[np.arange(batch_size), future_t]
            goals_cat = torch.cat([hindsight_goals, dg], 0)
            
            # Hindsight values
            discount = self.gamma ** (future_t - 1)
            values = to_torch((discount * 1).reshape(batch_size, 1))
            next_observation = self.unnorm(trajectories[:, 1, :self.observation_dim], 'observations')
            next_action = self.unnorm(trajectories[:, 1, self.observation_dim:], 'actions')
            next_q1, next_q2 = self.forward_target(next_observation, next_action, self.unnorm(goal_rand, 'goals'))
            td_target1 = torch.cat([values, self.gamma * next_q1], 0)
            td_target2 = torch.cat([values, self.gamma * next_q2], 0)
            
            # observation, action, next_observation, next_action, nextnext_observation = self.unnorm_transition(trajectories, self.has_object)
            # nextnext_action = ema_model(nextnext_observation, 
            #                         torch.ones_like(batch.rtgs), 
            #                         batch.goals[:, -1], self.has_object)[:, 0, self.observation_dim:]
            # observation_cat = next_observation.repeat(2,1)
            # action_cat = next_action.repeat(2,1)
            # goals = torch.cat([batch.goals[:, -1], goal_rand], 0)
            # goals = self.unnorm(goals, 'achieved_goals')
            
            # values = torch.ones((batch_size, 1)).to('cuda')
            # next_q1, next_q2 = self.forward_target(nextnext_observation, nextnext_action, goal_rand)
        else:
            # Hindsight goals
            ag = self.unnorm(batch.trajectories[:, :, :self.goal_dim], 'achieved_goals')
            dg = self.unnorm(goal_rand, 'goals')
            observation, action, next_observation, next_action, _ = self.unnorm_transition(trajectories, self.has_object)
            observation_cat = observation.repeat(2,1)
            action_cat = action.repeat(2,1)
            hindsight_goals = ag[np.arange(batch_size), 1]
            goals_cat = torch.cat([hindsight_goals, dg], 0)
            
            # Hindsight values
            values = torch.ones((batch_size, 1)).to('cuda')
            next_q1, next_q2 = self.forward_target(next_observation, next_action, self.unnorm(goal_rand, 'goals'))
            td_target1 = torch.cat([values, self.gamma * next_q1], 0)
            td_target2 = torch.cat([values, self.gamma * next_q2], 0)

        # calaulate q value
        pred_q1, pred_q2 = self.forward(observation_cat, action_cat, goals_cat)
        
        # For negative action reg
        # x = torch.cat([observation_cat, goals], -1)
        targ_q1, targ_q2 = self.forward_target(observation_cat, action_cat, goals_cat)
        targ_q = torch.min(targ_q1, targ_q2)
        # negative_action = self.actor(x)
        # min_q1_loss, min_q2_loss = self.forward(observation_cat, negative_action.detach(), goals)
        # sample negative action
        num_random_actions = 10
        random_actions = torch.FloatTensor(batch_size * num_random_actions, action.shape[-1]).uniform_(-1, 1).to(action.device)
        obs_rpt = observation.repeat_interleave(num_random_actions, axis=0)
        goals_rrpt = dg.repeat_interleave(num_random_actions, axis=0)
        rand_q1, rand_q2 = self.forward(obs_rpt, random_actions, goals_rrpt)        
        rand_q1 = rand_q1.reshape(batch_size, -1)
        rand_q2 = rand_q2.reshape(batch_size, -1)
        i1 = torch.distributions.Categorical(logits=rand_q1.detach()).sample()
        i2 = torch.distributions.Categorical(logits=rand_q2.detach()).sample()
        # x = torch.cat([observation, goals_rpt], -1)
        # negative_action = self.actor(x)
        
        # min_q1_loss, min_q2_loss = self.forward(observation, negative_action.detach(), goals_rpt)
        min_q1_loss = rand_q1[torch.arange(batch_size), i1].mean()
        min_q2_loss = rand_q2[torch.arange(batch_size), i2].mean()
        
        # Final loss
        loss1 = F.mse_loss(td_target1.detach(), pred_q1, reduction='mean') + (min_q1_loss**2).mean()
        loss2 = F.mse_loss(td_target2.detach(), pred_q2, reduction='mean') + (min_q2_loss**2).mean()
        
        return loss1, loss2, targ_q.mean(), (min_q1_loss**2).mean(), (min_q2_loss**2).mean()

    def loss_random(self, batch, goal_rand, ema_model):
        trajectories = batch.trajectories
        batch_size, horizon, _ = trajectories.shape
        
        # Hindsight goals
        if self.has_object:
            ag = batch.trajectories[:, :, self.goal_dim:2*self.goal_dim]
            her_indexes, t_indexes = self.make_indices(batch_size, horizon)
            
            observation = self.unnorm(trajectories[:, :, :self.observation_dim], 'observations')
            action = self.unnorm(trajectories[:, :, self.observation_dim:], 'actions')
            
            next_observation = self.unnorm(trajectories[:, 1:, :self.observation_dim], 'observations')
            next_observation = self.add_last_obs(next_observation)
            
            last_action = ema_model(next_observation[:, -1], 
                                    torch.ones_like(batch.rtgs), 
                                    ag[:, -1], self.has_object)[:, 0, self.observation_dim:]
            next_action = torch.cat([self.unnorm(trajectories[:, 1:, self.observation_dim:], 'actions'),
                                    last_action[:,None]], 1)
            
            # goals = to_np(batch.goals.clone())
            # np.random.shuffle(goals)
            goals = goal_rand[:, 0]
            goals[her_indexes] = ag[her_indexes, t_indexes].squeeze()
            goals = self.unnorm(goals, 'achieved_goals')
        else:
            ag = batch.trajectories[:, :, :self.goal_dim]
            her_indexes, t_indexes = self.make_indices(batch_size, horizon-1)
            
            observation = self.unnorm(trajectories[:, :-1, :self.observation_dim], 'observations')
            action = self.unnorm(trajectories[:, :-1, self.observation_dim:], 'actions')
            
            next_observation = self.unnorm(trajectories[:, 1:, :self.observation_dim], 'observations')
            next_action = self.unnorm(trajectories[:, 1:, self.observation_dim:], 'actions')

            # goals = to_np(batch.goals.clone())
            # np.random.shuffle(goals)
            goals = goal_rand[:, 0]
            goals[her_indexes] = ag[her_indexes, t_indexes].squeeze()
            goals = self.unnorm(goals, 'achieved_goals')
         
        
        # Hindsight values R
        h = horizon if 'Fetch' in self.env_name and self.has_object else horizon -1
        goals_rpt = einops.repeat(goals, 'b d -> b h d', h=h)
        next_q1, next_q2 = self.forward_target(next_observation, next_action, goals_rpt)
        td_target1 = self.gamma * next_q1
        td_target2 = self.gamma * next_q2
        td_target1[her_indexes, t_indexes] = 1
        td_target2[her_indexes, t_indexes] = 1
        
        
        # next_action_sampled = ema_model(next_observation, values, goal_rand)

        # calaulate q value
        pred_q1, pred_q2 = self.forward(observation, action, goals_rpt)
        targ_q1, targ_q2 = self.forward_target(observation, action, goals_rpt)
        targ_q = torch.min(targ_q1, targ_q2)
        
        # sample negative action
        num_random_actions = 10
        random_actions = torch.FloatTensor(batch_size * num_random_actions, h, action.shape[-1]).uniform_(-1, 1).to(action.device)
        obs_rpt = observation.repeat_interleave(num_random_actions, axis=0)
        goals_rrpt = goals_rpt.repeat_interleave(num_random_actions, axis=0)
        rand_q1, rand_q2 = self.forward(obs_rpt, random_actions, goals_rrpt)        
        rand_q1 = rand_q1.reshape(batch_size, -1)
        rand_q2 = rand_q2.reshape(batch_size, -1)
        i1 = torch.distributions.Categorical(logits=rand_q1.detach()).sample()
        i2 = torch.distributions.Categorical(logits=rand_q2.detach()).sample()
        # x = torch.cat([observation, goals_rpt], -1)
        # negative_action = self.actor(x)
        
        # min_q1_loss, min_q2_loss = self.forward(observation, negative_action.detach(), goals_rpt)
        min_q1_loss = rand_q1[torch.arange(batch_size), i1].mean()
        min_q2_loss = rand_q2[torch.arange(batch_size), i2].mean()
        
        loss1 = F.mse_loss(td_target1.detach(), pred_q1, reduction='mean') + (min_q1_loss**2).mean()
        loss2 = F.mse_loss(td_target2.detach(), pred_q2, reduction='mean') + (min_q2_loss**2).mean()
        
        return loss1, loss2, targ_q.mean(), (min_q1_loss**2).mean(), (min_q2_loss**2).mean()
    
    def unnorm(self, x, key):
        return to_torch(self.normalizer.unnormalize(to_np(x), key))

    def norm(self, x, key):
        return to_torch(self.normalizer(to_np(x), key))
    
    def unnorm_transition(self, trajectories, has_object):
        observation = self.unnorm(trajectories[:, 0, :self.observation_dim], 'observations')
        action = self.unnorm(trajectories[:, 0, self.observation_dim:], 'actions')
        next_observation = self.unnorm(trajectories[:, 1, :self.observation_dim], 'observations')
        next_action = self.unnorm(trajectories[:, 1, self.observation_dim:], 'actions')
        if has_object:
            nextnext_observation = self.last_obs(next_observation)
        else:
            nextnext_observation = next_observation[:, -1]

        return observation, action, next_observation, next_action, nextnext_observation

    def make_indices(self, batch_size, horizon, relabel_percent=0.5):
        her_indexes = np.where(np.random.uniform(size=batch_size) < relabel_percent)
        t_indexes = (np.random.randint(horizon, size=batch_size))[her_indexes]
        return her_indexes, t_indexes
    
    def last_obs(self, obs):
        last_obs = obs.clone()
        last_obs[:, :self.goal_dim] = obs[:, self.goal_dim:2*self.goal_dim]
        last_obs[:, 2*self.goal_dim:3*self.goal_dim] = torch.zeros_like(last_obs[:, 2*self.goal_dim:3*self.goal_dim])
        last_obs[:, 3*self.goal_dim:] = 0
        return last_obs
    

