import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from itertools import count
from collections import namedtuple

## Import function approximator for Q from other file
from Q_network import QNetwork
from whittle_network import WhittleNetwork

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
    def __init__(self, q_lr=0.01, whittle_lr=0.01, state_size=4, action_size=2, seed=123):
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
        self.whittle = WhittleNetwork(state_size=self.state_size).float().to(device)

        # optimizers
        self.optimizer_Q = optim.RMSprop(self.qnetwork.parameters(), lr=q_lr)
        self.optimizer_whittle = optim.RMSprop(self.whittle.parameters(), lr=whittle_lr)

        self.s0 = 2
        self.a0 = 0

    def Optimise_Q(self,Samples,trans):
        batch = self.Transition(*zip(*Samples))
        state_batch = torch.tensor(batch.state,dtype=torch.float32).unsqueeze(1).to(device)                 #[B,1]
        action_batch = torch.tensor(batch.action,dtype=torch.float32).unsqueeze(1).to(device)               #[B,1]
        reward_batch = torch.tensor(batch.reward,dtype=torch.float32).to(device)                            #[B]
        next_state_batch = torch.tensor(batch.next_state,dtype=torch.float32).unsqueeze(1).to(device)       #[B,1]
        len1 = action_batch.size()[0] #B
        with torch.no_grad():
            indices = self.whittle(state_batch).to(device) #[B,1]
            next_indices = self.whittle(next_state_batch).to(device) #[B,1]
            actions0 = torch.tensor([0.]* len1).unsqueeze(1).to(device) #[B,1]
            actions1 = torch.tensor([1.]* len1).unsqueeze(1).to(device) #[B,1]
            NSA_values0 = self.qnetwork(next_state_batch, actions0, next_indices) #[B]
            NSA_values1 = self.qnetwork(next_state_batch, actions1, next_indices) #[B]
            NSA_values = torch.max(NSA_values0, NSA_values1).to(device) #[B]
            SA_values = self.qnetwork(state_batch, action_batch, indices).to(device) #[B]
            fixed_Q_values = self.qnetwork(
                torch.tensor([self.s0]).unsqueeze(1).to(device),
                torch.tensor([self.a0]).unsqueeze(1).to(device),
                self.whittle(torch.tensor([self.s0]).unsqueeze(1).to(device))).to(device) #[1]
            targets = (1-action_batch.squeeze(1).to(torch.int64))*(reward_batch+indices.squeeze(1))+\
            action_batch.squeeze(1).to(torch.int64)*reward_batch + NSA_values - fixed_Q_values #[B]
            avg_part = targets-SA_values #[B]
            assert avg_part.requires_grad == False, "avg_part requires grad"
        # ================================================================
        
        # ========Calculating the gradient term =================
        state = torch.tensor([trans.state],dtype=torch.float32).unsqueeze(1).to(device) #[1,1]
        action = torch.tensor([trans.action],dtype=torch.float32).unsqueeze(1).to(device) #[1,1]
        reward = torch.tensor([trans.reward],dtype=torch.float32).to(device) #[1]
        next_state = torch.tensor([trans.next_state],dtype=torch.float32).unsqueeze(1).to(device) #[1,1]

        index = self.whittle(state).to(device) #[1,1]
        next_index = self.whittle(next_state).to(device) #[1,1]
        action0 = torch.tensor([0.]).unsqueeze(1).to(device) #[1,1]
        action1 = torch.tensor([1.]).unsqueeze(1).to(device) #[1,1]
        NSA_value0 = self.qnetwork(next_state, action0, next_index) #[1]
        NSA_value1 = self.qnetwork(next_state, action1, next_index) #[1]
        NSA_value = torch.max(NSA_value0, NSA_value1).to(device) #[1]
        SA_value = self.qnetwork(state, action, index).to(device) #[1]
        fixed_Q_value = self.qnetwork(
            torch.tensor([self.s0]).unsqueeze(1).to(device),
            torch.tensor([self.a0]).unsqueeze(1).to(device),
            self.whittle(torch.tensor([self.s0]).unsqueeze(1).to(device))).to(device) #[1]
        target = (1-action.squeeze(1).to(torch.int64))*(reward+index.squeeze(1))+\
            action.squeeze(1).to(torch.int64)*reward + NSA_value - fixed_Q_value #[1]
        assert target.requires_grad == True, "target doesn't requires grad"
        # ================================================================
        avg_part = avg_part+(target-SA_value).detach() #[B+1]
        # tensor.detach() creates a tensor that shares storage with tensor that does not require grad.
        avg_part = torch.mean(avg_part) #[]
        assert avg_part.requires_grad == False, "avg_part requires grad"
        # ================================================================
        loss = torch.mul(avg_part,target-SA_value) #[1]
        self.optimizer_Q.zero_grad()
        self.optimizer_whittle.zero_grad()
        loss.backward()
        for param in self.qnetwork.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_Q.step()

        actual_loss = F.mse_loss(SA_value,target)
        return actual_loss.item()


    def Optimise_whittle(self,Samples,trans):
        batch = self.Transition(*zip(*Samples))
        state_batch = torch.tensor(batch.state,dtype=torch.float32).unsqueeze(1).to(device)                 #[B,1]
        action_batch = torch.tensor(batch.action,dtype=torch.float32).unsqueeze(1).to(device)               #[B,1]
        reward_batch = torch.tensor(batch.reward,dtype=torch.float32).to(device)                            #[B]
        next_state_batch = torch.tensor(batch.next_state,dtype=torch.float32).unsqueeze(1).to(device)       #[B,1]
        len1 = action_batch.size()[0] #B
        with torch.no_grad():
            indices = self.whittle(state_batch).to(device) #[B,1]
            next_indices = self.whittle(next_state_batch).to(device) #[B,1]
            actions0 = torch.tensor([0.]* len1).unsqueeze(1).to(device) #[B,1]
            actions1 = torch.tensor([1.]* len1).unsqueeze(1).to(device) #[B,1]
            NSA_values0 = self.qnetwork(next_state_batch, actions0, next_indices) #[B]
            NSA_values1 = self.qnetwork(next_state_batch, actions1, next_indices) #[B]
            NSA_values = torch.max(NSA_values0, NSA_values1).to(device) #[B]
            Q_values = self.qnetwork(state_batch, actions1, indices).to(device) #[B]
            fixed_Q_values = self.qnetwork(
                torch.tensor([self.s0]).unsqueeze(1).to(device),
                torch.tensor([self.a0]).unsqueeze(1).to(device),
                self.whittle(torch.tensor([self.s0]).unsqueeze(1).to(device))).to(device) #[1]
            targets = Q_values-reward_batch+fixed_Q_values-NSA_values #[B]
            avg_part = targets-indices.squeeze(1) #[B]
            assert avg_part.requires_grad == False, "avg_part requires grad"
        # ================================================================
        
        # ========Calculating the gradient term =================
        state = torch.tensor([trans.state],dtype=torch.float32).unsqueeze(1).to(device) #[1,1]
        action = torch.tensor([trans.action],dtype=torch.float32).unsqueeze(1).to(device) #[1,1]
        reward = torch.tensor([trans.reward],dtype=torch.float32).to(device) #[1]
        next_state = torch.tensor([trans.next_state],dtype=torch.float32).unsqueeze(1).to(device) #[1,1

        index = self.whittle(state).to(device) #[1,1]
        next_index = self.whittle(next_state).to(device) #[1,1]
        action0 = torch.tensor([0.]).unsqueeze(1).to(device) #[1,1]
        action1 = torch.tensor([1.]).unsqueeze(1).to(device) #[1,1]
        NSA_value0 = self.qnetwork(next_state, action0, next_index) #[1]
        NSA_value1 = self.qnetwork(next_state, action1, next_index) #[1]
        NSA_value = torch.max(NSA_value0, NSA_value1).to(device) #[1]
        Q_value = self.qnetwork(state, action1, index).to(device) #[1]
        fixed_Q_value = self.qnetwork(
            torch.tensor([self.s0]).unsqueeze(1).to(device),
            torch.tensor([self.a0]).unsqueeze(1).to(device),
            self.whittle(torch.tensor([self.s0]).unsqueeze(1).to(device))).to(device) #[1]
        target = Q_value-reward+fixed_Q_value-NSA_value #[1]
        assert target.requires_grad == True, "target doesn't requires grad"
        # ================================================================
        avg_part = avg_part+(target-index.squeeze(1)).detach() #[B+1]
        # tensor.detach() creates a tensor that shares storage with tensor that does not require grad.
        avg_part = torch.mean(avg_part) #[]
        assert avg_part.requires_grad == False, "avg_part requires grad"
        # ================================================================
        loss = torch.mul(avg_part,target-index.squeeze(1)) #[1]
        self.optimizer_whittle.zero_grad()
        self.optimizer_Q.zero_grad()
        loss.backward()
        for param in self.qnetwork.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_whittle.step()

        actual_loss = F.mse_loss(index.squeeze(1),target)
        return actual_loss.item()
        

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