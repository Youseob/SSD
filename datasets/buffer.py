# 
import numpy as np

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

def discount_cumsum(x, gamma=1.):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

class ReplayBuffer:

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=np.int),
            'rtgs': np.zeros((max_n_episodes, max_path_length, 1), dtype=np.float32),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path):
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length

        ## if first path added, set keys based on contents
        self._add_keys(path)

        ## add tracked keys in path
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict: self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        ## penalize early termination
        if path['terminals'].any() and self.termination_penalty is not None:
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty

        ## record path length
        self._dict['path_lengths'][self._count] = path_length
        
        ## record rtg
        rtg = atleast_2d(discount_cumsum(path['rewards']))
        self._dict['rtgs'][self._count, :path_length] = rtg
        
        ## increment path counter
        self._count += 1
        
        
    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        ## remove extra slots
        for key in self.keys + ['path_lengths', 'rtgs', 'goals']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')


class HERReplayBuffer(ReplayBuffer):
        
    def add_path(self, path):
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length
        
        self._add_keys(path)
        
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict: self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array
        
        ## add hindsight goal and reward
        if 'goals' not in self._dict: self._allocate('goals', array)
        idx=0
        for i in range(path_length):
            if (self._dict['infos/goal'][self._count, i] != self._dict['infos/goal'][self._count, i+1]).all():
                self._dict['goals'][self._count, idx:i+1] = atleast_2d(path['observations'])[i]
                self._dict['rewards'][self._count, i] = 1
                idx = i+1
        self._dict['goals'][self._count, idx:path_length] = atleast_2d(path['observations'])[-1]
        
        ## penalize early termination
        if path['terminals'].any() and self.termination_penalty is not None:
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty
            
        ## record path length
        self._dict['path_lengths'][self._count] = path_length
        
        ## add rtg
        rtg = atleast_2d(discount_cumsum(path['rewards']))
        self._dict['rtgs'][self._count, :path_length] = rtg
        
        self._count += 1