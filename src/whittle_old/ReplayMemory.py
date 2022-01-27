from collections import namedtuple
from collections import deque
import random
import torch


random.seed(30)
torch.set_default_dtype(torch.float32)


class ReplayMemory(object):
    def __init__(self,capacity,batch_size):
        self.capacity = capacity
        self.memory =  deque(maxlen=self.capacity)
        self.batch_size = batch_size

    def Push_transition(self,transitions):
        """
        Recieves a batch of transitions
        """
        for trans in transitions:
          self.memory.append(trans)

    def Sample_batch_train(self):
        # returns a batch (size batch_size B) of transitions sampled randomly
        if len(self.memory)<self.batch_size:
            return None
        return random.sample(self.memory,self.batch_size)

    def Sample_batch_FG(self,s1,a1):
        # returns a batch of transitions of size B with same state and action
        # sampled randomly
        temp = [self.memory[i] for i in range(len(self.memory)) if (self.memory[i].state,self.memory[i].action)==(s1,a1)]
        if len(temp)<self.batch_size:
            return None
        return random.sample(temp,self.batch_size)

    def Sample_batch_whittle(self):
        # returns a batch of transitions of size B with action = 0
        # sampled randomly
        temp = [self.memory[i] for i in range(len(self.memory)) if self.memory[i].action==0]
        if len(temp)<self.batch_size:
            return None
        return random.sample(temp,self.batch_size)

    def __len__(self):
        # returns size of the memory
        return len(self.memory)

    def print_memory(self):
        # prints the content of the replay buffer
        print(self.memory)