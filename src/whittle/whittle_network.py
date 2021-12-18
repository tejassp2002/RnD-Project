import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Whittle(torch.nn.Module):
    def __init__(self):
        """
        Polynomially parameterized function for
        Whittle Indices calculation
        """
        super().__init__()
        # parameters -> z0, z1, z2, z3
        self.powers = 4
        self.parameter = torch.nn.Parameter(torch.zeros(1,self.powers))

    def basis(self,state):
      # returns tensor([[s^0],[s^1],[s^2],[s^3]]) size of tensor [4,1]
      # states are from {1,2,3,4}
      if not torch.is_tensor(state):
        state = torch.Tensor([state])
      x = torch.zeros(self.powers,1)
      for i in range(self.powers):
        x[i][0]=torch.pow(state,i)
      return x.to(device)

    def whittle_index(self,state):
      # returns whittle index calculated as
      # index = z0*s^0+z1*s^1+z2*s^2+z3*s^3
      index = torch.matmul(self.parameter, self.basis(state)).squeeze(1)
      return index

    def whittle_indices(self,states):
      # calculates whittle indices for given states
      indices = self.whittle_index(states[0])
      for i in range(1,len(states)):
        indices = torch.cat((indices,self.whittle_index(states[i])))
      return indices



class New_Whittle(torch.nn.Module):
    def __init__(self,state_size=4,fc1_unit = 256):
        super(New_Whittle, self).__init__()
        self.state_size = state_size
        self.fc1 = nn.Linear(self.state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,1)
        self.apply(self.init_weights)

    def forward(self,states):
        # states shape: [B,1]
        states = self.one_hot(states)               #[B,4]
        hidden = F.relu(self.fc1(states))
        indices = self.fc2(hidden)                  #[B,1]
        return indices.squeeze(1)                   #[B]

    def init_weights(self,m):
        """
        Initialized the weights for all Linear Layers
        """
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)

    def one_hot(self,states):
        """
        One hot encoding of states
        1 -> [1,0,0,0]   2 -> [0,1,0,0]
        3 -> [0,0,1,0]   4 -> [0,0,0,1]
        """
        #state_batch shape: [B,1]
        return F.one_hot(states.to(torch.int64)-1,self.state_size).squeeze(1).to(torch.float32)