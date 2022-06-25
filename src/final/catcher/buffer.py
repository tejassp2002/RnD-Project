from typing import Any, Dict, Generator, List, Optional, Union, Callable, NamedTuple, Tuple
import numpy as np
import torch as th
from utils import BaseBuffer
import random

class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    rewards: th.Tensor

class FGReplayBuffer(BaseBuffer):
    def __init__(
        self,
        obs,
        buffer_size: int,
        device: Union[th.device, str] = "cpu",
    ):
        super().__init__(obs, buffer_size, device)  
        self.obs_shape = obs.shape

        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32)  
        self.next_observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32)  
        self.actions = np.zeros((self.buffer_size,1), dtype=np.float32)  
        self.rewards = np.zeros((self.buffer_size,1), dtype=np.float32)         

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray
    ) -> None:

        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds, :],
            self.next_observations[batch_inds, :],
            self.rewards[batch_inds]
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def sampleFG(self, obs, act,batch_size) -> ReplayBufferSamples:
        '''
        sample all the transitions with the given state and action pair
        '''
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = [i for i in range(upper_bound) if \
            th.equal(obs,th.tensor(self.observations[i, :]).to(self.device))\
                      and th.equal(act,th.tensor(self.actions[i, :]).to(self.device))]

        if len(batch_inds)>batch_size:
            batch_inds = random.sample(batch_inds,batch_size)
        
        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds, :],
            self.next_observations[batch_inds, :],
            self.rewards[batch_inds]
        )
        
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
