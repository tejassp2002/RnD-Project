from collections import namedtuple
from collections import deque
import random
import torch


random.seed(30)
torch.set_default_dtype(torch.float32)


class ReplayMemory(object):
    def __init__(self,capacity,batch_size=30):
        self.capacity = capacity
        self.memory =  deque(maxlen=self.capacity)
        self.batch_size = batch_size

    def push_transition(self,transitions):
        """
        Recieves a batch of transitions
        """
        for trans in transitions:
          self.memory.append(trans)

    def sample_trans_Q(self):
        # returns a single transition sampled randomly from the replay buffer
        return random.sample(self.memory,1)
    
    def sample_trans_whittle(self):
        # returns a single transition sampled randomly from the replay buffer
        temp = [self.memory[i] for i in range(len(self.memory)) if 
            self.memory[i].action==0]
        return random.sample(temp,1)

    def sample_batch(self,trans):
        # returns a batch of transitions of size B or less than B 
        # with same state and action, sampled randomly
        temp = [self.memory[i] for i in range(len(self.memory)) if 
            (self.memory[i].state,self.memory[i].action)==(trans.state,trans.action)]
        temp.remove(trans)
        if len(temp)<self.batch_size:
            return temp
        return random.sample(temp,self.batch_size)

    def __len__(self):
        # returns size of the memory
        return len(self.memory)

    def print_memory(self):
        # prints the content of the replay buffer
        print(self.memory)