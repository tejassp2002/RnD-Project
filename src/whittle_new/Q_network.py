import torch
import torch.nn as nn
#import numpy as np
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

class QNetwork(nn.Module):
    def __init__(self,state_size,fc1_unit = 2048):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): number of states
            fc1_unit (int): Number of nodes in first hidden layer
        """
        super(QNetwork, self).__init__()  ## calls __init__ method of nn.Module class
        self.state_size = state_size
        self.fc1 = nn.Linear(3, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, 1)

    def forward(self, state_batch, action_batch, index_batch):
        """
        Given state, action, and whittle index outputs Q-value
        [s,a,lambda] -> Q(s,a)
        Shape: 
        state_batch: [B,1]; action_batch: [B,1]; index_batch: [B,1]
        returns Q value of size [B]
        """
        # convert the state b/w [1,2,..,state_size] to [0,1]
        state_batch = state_batch/self.state_size #[B,1]
        input_batch = torch.cat((state_batch,action_batch,index_batch),1) #[B,3]
        hidden = F.relu(self.fc1(input_batch))
        q_values = self.fc2(hidden)                    #[B,1]
        return q_values.squeeze(1)
