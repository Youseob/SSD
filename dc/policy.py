import torch
import os
import numpy as np

from utils.arrays import to_torch, to_np

class GoalTorqueControl:
    def __init__(self, ema_model, normalizer, observation_dim, goal_dim):
        self.action_list = []
        self.ema_model = ema_model
        self.normalizer = normalizer
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
    
    def act(self, state, condition, target, at_goal):
        if at_goal:
            action = target - state[:self.goal_dim] - state[self.goal_dim:]
            self.action_list = []
        else:
            if len(self.action_list) == 0:
                normed_state = to_torch(self.normalizer(state, 'observations')).reshape(1, self.observation_dim)
                normed_target = to_torch(self.normalizer(target, 'goals')).reshape(1, self.goal_dim)
                samples = self.ema_model(normed_state, condition, normed_target)
                self.action_list = self.normalizer.unnormalize(to_np(samples)[0, :, self.observation_dim:], 'actions')
            action = self.action_list[0]
            self.action_list = np.delete(self.action_list, 0, 0)
        return action

class ConditionControl:
    def __init__(self, ema_model, normalizer, observation_dim, goal_dim, horizon, gamma):
        self.action_list = []
        self.ema_model = ema_model
        self.normalizer = normalizer
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        self.horizon = horizon
        self.gamma = gamma
        self.threshold = np.linalg.norm((normalizer.normalizers['observations'].maxs - \
                                        normalizer.normalizers['observations'].mins)[:goal_dim]) * 0.3
    
    # def far(self, state, target):
    #     if np.linalg.norm(state[:self.goal_dim] - target) <= self.threshold:
    #         return False
    #     else:
    #         return True
    
    def act(self, state, condition, target, at_goal):
        if at_goal:
            action = target - state[:self.goal_dim] - state[self.goal_dim:]
            self.action_list = []
        else:
            # if not self.far(state, target):
                if len(self.action_list) == 0:
                    normed_state = to_torch(self.normalizer(state, 'observations')).reshape(1, self.observation_dim)
                    normed_target = to_torch(self.normalizer(target, 'goals')).reshape(1, self.goal_dim)
                    samples = self.ema_model(normed_state, condition, normed_target)
                    self.action_list = self.normalizer.unnormalize(to_np(samples)[0, :, self.observation_dim:], 'actions')
                action = self.action_list[0]
                self.action_list = np.delete(self.action_list, 0, 0)
            # else:
            #     if len(self.action_list) == 0:
            #         self.target_tmp = target
            #         while self.far(state, self.target_tmp):
            #             self.target_tmp = self.env.observation_space.sample()[:self.goal_dim]                
            #             if not self.good_surrogate(state, target):
            #                 self.target_tmp = target
            #         condition = condition * self.gamma ** self.horizon
            #         normed_state = to_torch(self.normalizer(state, 'observations')).reshape(1, self.observation_dim)
            #         normed_target = to_torch(self.normalizer(target, 'goals')).reshape(1, self.goal_dim)
            #         samples = self.ema_model(normed_state, condition, normed_target)
            #         self.action_list = self.normalizer.unnormalize(to_np(samples)[0, :, self.observation_dim:], 'actions')
                action = self.action_list[0]
                self.action_list = np.delete(self.action_list, 0, 0)
            
        return action

class GoalPositionControl:
    def __init__(self, ema_model, normalizer, observation_dim, goal_dim, p_gain=10., d_gain=-1.):
        self.next_waypoint_list = []
        self.ema_model = ema_model
        self.normalizer = normalizer
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        self.p_gain = p_gain
        self.d_gain = d_gain
    
    def act(self, state, condition, target, at_goal):
        if at_goal:
            action = self.p_gain * (target - state[:self.goal_dim]) + self.d_gain * state[self.goal_dim:]
            self.next_waypoint_list = []
        else:
            if len(self.next_waypoint_list) == 0:
                normed_state = to_torch(self.normalizer(state, 'observations')).reshape(1, self.observation_dim)
                normed_target = to_torch(self.normalizer(target, 'goals')).reshape(1, self.goal_dim)
                samples = self.ema_model(normed_state, condition, normed_target)
                self.next_waypoint_list = self.normalizer.unnormalize(to_np(samples)[0, 1:, :self.observation_dim], 'observations')
            # action = self.next_waypoint_list[0, :self.goal_dim] - state[:self.goal_dim] \
            #         + self.next_waypoint_list[0, self.goal_dim:] - state[self.goal_dim:]
            action = self.p_gain * (self.next_waypoint_list[0, :self.goal_dim] - state[:self.goal_dim]) \
                    + self.d_gain * state[self.goal_dim:]
            self.next_waypoint_list = np.delete(self.next_waypoint_list, 0, 0)
        
        return action

class SampleEveryControl:
    def __init__(self, ema_model, normalizer, observation_dim, goal_dim):
        self.ema_model = ema_model
        self.normalizer = normalizer
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        
    def act(self, state, condition, target, at_goal=None):
        normed_state = to_torch(self.normalizer(state, 'observations')).reshape(1, self.observation_dim)
        normed_target = to_torch(self.normalizer(target, 'goals')).reshape(1, self.goal_dim)
        samples = self.ema_model(normed_state, condition, normed_target)
        action = self.normalizer.unnormalize(to_np(samples)[0, 0, self.observation_dim:], 'actions')
        
        return action