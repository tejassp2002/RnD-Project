import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WhittleNetwork(torch.nn.Module):
    def __init__(self,state_size,fc1_unit = 1024):
        super(WhittleNetwork, self).__init__()
        self.state_size = state_size
        self.fc1 = nn.Linear(1, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,1)
        # self.apply(self.init_weights)

    def forward(self,state_batch):
        """
        Shape: 
        input- state_batch: [B,1]; output- indices: [B,1]
        """
        # convert the state b/w [1,2,..,state_size] to [0,1]
        state_batch = state_batch/self.state_size #[B,1]
        hidden = F.relu(self.fc1(state_batch))
        indices = self.fc2(hidden)                  #[B,1]
        return indices  #[B,1]

    def init_weights(self,m):
        """
        Initialized the weights for all Linear Layers
        """
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)