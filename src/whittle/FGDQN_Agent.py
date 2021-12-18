import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from itertools import count
from collections import namedtuple

## Import function approximator for Q from other file
from Q_network import QNetwork
from whittle_network import Whittle
from whittle_network import New_Whittle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


torch.autograd.set_detect_anomaly(True)
# Set up matplotlib

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)



class FGDQN_agent:
    def __init__(self, q_lr=0.01, whittle_lr=0.01, state_size=4, action_size=2, seed=123, base_lr=0.01):
        """ Initialize an Agent object.
        ======
        Params
        ======
            state_size (int): number of states
            action_size (int): number of actions
            seed (int): random seed
        """

        self.action_size = action_size
        self.state_size = state_size
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.seed = np.random.seed(seed)

        # Initialises the Q-Network
        self.qnetwork = QNetwork(state_size=self.state_size).float().to(device)
        # Initialises the Whittle-Network

        # self.whittle = Whittle().float().to(device)
        self.whittle = New_Whittle(state_size=self.state_size).float().to(device)

        # optimizers
        self.optimizer_Q = optim.RMSprop(self.qnetwork.parameters(), lr=q_lr)
        self.optimizer_whittle = optim.RMSprop(self.whittle.parameters(), lr=whittle_lr)

        self.fixed_state = torch.tensor([1],dtype=torch.float32).unsqueeze(1).to(device)

    def Optimise_Q(self,Samples):
        # Transpose the batch. This converts batch-array of Transitions to Transition of batch-arrays.
        # converts batch of transitions to transitions of batches.
        # [(s1,a1,r1,s'1),...,(sn,an,rn,s'n)] -> [(s1,..,sn),(a1,..,an),(r1,..,rn),(s'1,..,s'n)]
        batch = self.Transition(*zip(*Samples))

        state_batch = torch.tensor(batch.state,dtype=torch.float32).unsqueeze(1).to(device)                 #[B,1]
        action_batch = torch.tensor(batch.action,dtype=torch.float32).unsqueeze(1).to(device)               #[B,1]
        reward_batch = torch.tensor(batch.reward,dtype=torch.float32).to(device)                            #[B]
        next_state_batch = torch.tensor(batch.next_state,dtype=torch.float32).unsqueeze(1).to(device)       #[B,1]

        #torch.max returns dictionary containing {max value, index of max value}
        NSA_values = torch.max(self.qnetwork(state_batch),1)[0]                                             #[B]

        # with torch.gather we get all Q values corresponding to the actions in action_batch
        SA_values = self.qnetwork(state_batch).gather(1, action_batch.to(torch.int64)).squeeze(1)           #[B]

        #taking reference value as max Q value of arbitrary but fixed state
        fixed_Q = self.qnetwork(self.fixed_state)
        fixed_Q = torch.max(fixed_Q,1)[0] 

        with torch.no_grad():
            #indices = self.whittle.whittle_indices(state_batch.squeeze(1)).to(device)                      #[B]
            indices = self.whittle(state_batch).to(device)                                                  #[B]

        actions = action_batch.squeeze(1)

        labels = (1-actions.to(torch.int64))*(reward_batch+indices)+\
            actions*reward_batch+NSA_values-fixed_Q                                          #[B]

        labels = labels.to(torch.float32)
        loss = F.mse_loss(SA_values,labels)

        self.optimizer_Q.zero_grad()
        loss.backward()

        #clipping gradient to stabilize learning
        for param in self.qnetwork.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer_Q.step()
        return loss.item()


    def Optimise_whittle(self,Samples):
        # Transpose the batch. This converts batch-array of Transitions to Transition of batch-arrays.
        # converts batch of transitions to transitions of batches.
        # [(s1,a1,r1,s'1),...,(sn,an,rn,s'n)] -> [(s1,..,sn),(a1,..,an),(r1,..,rn),(s'1,..,s'n)]

        batch = self.Transition(*zip(*Samples))

        state_batch = torch.tensor(batch.state,dtype=torch.float32).unsqueeze(1).to(device)                 #[B,1]
        action_batch = torch.tensor(batch.action,dtype=torch.float32).unsqueeze(1).to(device)               #[B,1]
        reward_batch = torch.tensor(batch.reward,dtype=torch.float32).to(device)                            #[B]
        next_state_batch = torch.tensor(batch.next_state,dtype=torch.float32).unsqueeze(1).to(device)       #[B,1]

        with torch.no_grad():
            Q_value = self.qnetwork(state_batch)[:,1]
            NSA_values = torch.max(self.qnetwork(state_batch),1)[0]                                         #[B]
            fixed_Q = self.qnetwork(self.fixed_state)
            fixed_Q = torch.max(fixed_Q,1)[0]
            labels = Q_value-reward_batch-NSA_values+fixed_Q

        # indices = self.whittle.whittle_indices(state_batch.squeeze(1)).to(device)                         #[B]
        indices = self.whittle(state_batch).to(device)                                                      #[B]

        loss = F.mse_loss(indices, labels)

        self.optimizer_whittle.zero_grad()
        loss.backward()

        for param in self.whittle.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer_whittle.step()

        return loss.item()

    def get_indices(self):
        x = []
        for i in range(1,self.state_size+1):
            x.append(round(self.whittle(torch.tensor([i]).unsqueeze(1).to(device)).item(),3))
        return x

    def get_Q(self,state):
        with torch.no_grad():
            action0 = torch.tensor([0.]).unsqueeze(1).to(device)
            action1 = torch.tensor([1.]).unsqueeze(1).to(device)
            Q0 = self.qnetwork(state.unsqueeze(1).to(device),action0)
            Q1 = self.qnetwork(state.unsqueeze(1).to(device),action1)
            best_action = torch.tensor([0.]) if Q0>=Q1 else torch.tensor([1.])
        assert best_action.requires_grad == False, "something is wrong"
        #print("best action", best_action)
        return best_action.to(torch.device("cpu"))

    def get_Q_values(self):
        with torch.no_grad():
            Q = []
            for s in range(1,self.state_size+1):
                A = self.qnetwork(torch.tensor([s],dtype=torch.float32).unsqueeze(1).to(device)).squeeze(0)
                Q.append(A.cpu().detach().numpy())
        return Q