import torch
import os
import numpy as np

from utils.arrays import to_torch, to_np

class GoalTorqueControl:
    def __init__(self, ema_model, normalizer, observation_dim, goal_dim, has_object):
        self.action_list = []
        self.ema_model = ema_model
        self.normalizer = normalizer
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        self.has_object = has_object
    
    def act(self, state, condition, target, at_goal):
        if at_goal:
            action = target - state[:self.goal_dim] - state[self.goal_dim:]
            self.action_list = []
        else:
            if len(self.action_list) == 0:
                normed_state = to_torch(self.normalizer(state, 'observations')).reshape(1, self.observation_dim)
                normed_target = to_torch(self.normalizer(target, 'goals')).reshape(1, self.goal_dim)
                samples = self.ema_model(normed_state, condition, normed_target, self.has_object)
                self.action_list = self.normalizer.unnormalize(to_np(samples)[0, :, self.observation_dim:], 'actions')
            action = self.action_list[0]
            self.action_list = np.delete(self.action_list, 0, 0)
        return action
    
class GoalPositionControl:
    def __init__(self, ema_model, normalizer, observation_dim, goal_dim, has_object, p_gain=10., d_gain=-1.):
        self.next_waypoint_list = []
        self.ema_model = ema_model
        self.normalizer = normalizer
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.has_object = has_object
    
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
                    + self.d_gain * state[self.goal_dim:2*self.goal_dim]
            self.next_waypoint_list = np.delete(self.next_waypoint_list, 0, 0)
        
        return action

class DDPositionControl:
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
                returns = torch.cat([normed_target, condition], -1)
                samples = self.ema_model(normed_state, None, returns)
                self.next_waypoint_list = self.normalizer.unnormalize(to_np(samples)[0, 1:, :self.observation_dim], 'observations')
            # action = self.next_waypoint_list[0, :self.goal_dim] - state[:self.goal_dim] \
            #         + self.next_waypoint_list[0, self.goal_dim:] - state[self.goal_dim:]
            action = self.p_gain * (self.next_waypoint_list[0, :self.goal_dim] - state[:self.goal_dim]) \
                    + self.d_gain * state[self.goal_dim:]
            self.next_waypoint_list = np.delete(self.next_waypoint_list, 0, 0)
        
        return action
    
class SampleEveryControl:
    def __init__(self, ema_model, normalizer, observation_dim, goal_dim, has_object):
        self.ema_model = ema_model
        self.normalizer = normalizer
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        self.has_object = has_object
        
    def act(self, state, condition, target, at_goal=None):
        normed_state = to_torch(self.normalizer(state, 'observations')).reshape(1, self.observation_dim)
        normed_target = to_torch(self.normalizer(target, 'goals')).reshape(1, self.goal_dim)
        samples = self.ema_model(normed_state, condition, normed_target, self.has_object)
        action = self.normalizer.unnormalize(to_np(samples)[0, 0, self.observation_dim:], 'actions')
        
        return action
    
class FetchControl:
    def __init__(self, ema_model, normalizer, observation_dim, goal_dim, has_object):
        self.ema_model = ema_model
        self.normalizer = normalizer
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        self.has_object = has_object
        
    def act(self, state, condition, target, at_goal=None):
        if at_goal:
            action = np.zeros((4,))
            # action[:3] = target - state[:self.goal_dim]
        else:
            normed_state = to_torch(self.normalizer(state, 'observations')).reshape(1, self.observation_dim)
            normed_target = to_torch(self.normalizer(target, 'achieved_goals')).reshape(1, self.goal_dim)
            samples = self.ema_model(normed_state, condition, normed_target, self.has_object)
            action = self.normalizer.unnormalize(to_np(samples)[0, 0, self.observation_dim:], 'actions')
        
        return action