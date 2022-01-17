import torch
import torch.nn as nn
#import numpy as np
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size, action_size,fc1_unit = 2000, fc2_unit = 8):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()  ## calls __init__ method of nn.Module class
        self.fc1 = nn.Linear(2, fc1_unit)
        #self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc2 = nn.Linear(fc1_unit, 1)

    def forward(self, state_batch,action_batch):
        #state_batch shape: [B,1]
        input_x = torch.cat([state_batch,action_batch],dim=1)
        #print("state_inuput",input_x)
        x_state = F.relu(self.fc1(input_x))
        #x_state = F.relu(self.fc2(x_state))
        q_values = self.fc2(x_state)
        #out shape is [B,1]
        #returning shape is [B,1]
        return q_values.squeeze(1)
