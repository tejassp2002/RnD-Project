import torch
import torch.nn as nn
#import numpy as np
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

class Network(nn.Module):
    def __init__(self,state_size):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_size,1024),nn.ReLU())
                                            # nn.Linear(1024,1024),nn.ReLU())
        self.whittle = nn.Linear(1024,1)
        self.q_network = nn.Linear(1024,2)
        self.state_size = state_size
    
    def forward(self,state_batch):
        state_batch = self.one_hot(state_batch)         #[B,4]
        hidden = self.shared(state_batch)               #[B,1024]
        whittle = self.whittle(hidden)                  #[B,1]
        q_values = self.q_network(hidden)               #[B,2]
        return q_values, whittle

    def one_hot(self,states):
        """
        One hot encoding of states
        1 -> [1,0,0,0]   2 -> [0,1,0,0]
        3 -> [0,0,1,0]   4 -> [0,0,0,1]
        """
        #state_batch shape: [B,1]
        return F.one_hot(states.to(torch.int64)-1,self.state_size).squeeze(1).to(torch.float32)
