#######################################################################################################################
#                                                                                                                     #
#    Kishor Patil                                                                                                     #
#    @ Oct, 2020; Reinforcement Learning                                                                              #
#    Web page crawling; Whittle indices;                                                                              #
#    Deep learning Framework                                                                                          #
#    Replay Memory #only 1 sample at the moment from replay memory                                                    #
#######################################################################################################################

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
        data.remove(trans)
        return data 

    def __len__(self):
        return len(self.memory)

    def print_memory(self):
        print(self.memory)








