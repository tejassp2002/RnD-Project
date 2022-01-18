from collections import namedtuple
from collections import deque
import random
import torch
import numpy as np


random.seed(30)
torch.set_default_dtype(torch.float32)

Transition = namedtuple('Transition', ('state', 'action', 'reward','next_state'))
class ReplayMemory(object):
    def __init__(self,capacity,batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory =  deque(maxlen=self.capacity)

    def push_transition(self,arm_states, current_action, immediate_reward, next_state):
        self.memory.append(Transition(arm_states, current_action, immediate_reward, next_state))

    def sample_batch_train(self):
        return random.sample(self.memory,self.batch_size)

    def sample_batch_FG(self,trans):
        data = [self.memory[i] for i in range(len(self.memory)) if 
            (self.memory[i].state,self.memory[i].action)==(trans.state,trans.action)]
        data.remove(trans) # removing the current transition from the batch
        if len(data)<self.batch_size:
            return data
        else:
            return random.sample(data,self.batch_size)

    def __len__(self):
        return len(self.memory)

    def print_memory(self):
        print(self.memory)








