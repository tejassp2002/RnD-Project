import torch
import torch.nn as nn
#import numpy as np
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self,state_size=4,fc1_unit = 1024, fc2_unit = 256):
        """
        Initialize parameters and build model.
        Params
        =======
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()  ## calls __init__ method of nn.Module class
        self.state_size = state_size
        self.fc1 = nn.Linear(self.state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, 2)

    def forward(self, state_batch):
        """
        Given state it computers Q(s,a) for all a in A
        returns Q value of size [B,2]
        """
        # state_batch shape: [B,1]
        state_batch = self.one_hot(state_batch)        #[B,4]
        hidden = F.relu(self.fc1(state_batch))
        hidden = F.relu(self.fc2(hidden))
        q_values = self.fc3(hidden)                    #[B,2]
        return q_values

    def one_hot(self,states):
        """
        One hot encoding of states
        1 -> [1,0,0,0]   2 -> [0,1,0,0]
        3 -> [0,0,1,0]   4 -> [0,0,0,1]
        """
        #state_batch shape: [B,1]
        return F.one_hot(states.to(torch.int64)-1,self.state_size).squeeze(1).to(torch.float32)