import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from itertools import count
from collections import namedtuple

## Import function approximator for Q from other file
from Q_network import Network
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
        self.network = Network(state_size=self.state_size).float().to(device)
        # Initialises the Whittle-Network

        # optimizers   
        self.optimizer_Q = optim.Adam([
            {"params": self.network.shared.parameters(), "lr": 1e-3},
            # {"params": self.network.q_layer.parameters(), "lr": 1e-3},
            {"params": self.network.q_network.parameters(), "lr": 1e-3}])
        self.optimizer_whittle = optim.Adam([
            {"params": self.network.shared.parameters(), "lr": 1e-3},
            {"params": self.network.whittle.parameters(), "lr": 1e-3}])

        self.s0 = 2
        self.a0 = 0

    def Optimise_Q(self,Samples,trans):
        batch = self.Transition(*zip(*Samples))
        state_batch = torch.tensor(batch.state,dtype=torch.float32).unsqueeze(1).to(device)                 #[B,1]
        action_batch = torch.tensor(batch.action,dtype=torch.int64).unsqueeze(1).to(device)                 #[B,1]
        reward_batch = torch.tensor(batch.reward,dtype=torch.float32).unsqueeze(1).to(device)               #[B,1]
        next_state_batch = torch.tensor(batch.next_state,dtype=torch.float32).unsqueeze(1).to(device)       #[B,1]
        with torch.no_grad():
            SA_values, indices = self.network(state_batch) #[B,2], [B,1]
            SA_values = SA_values.gather(1, action_batch).to(device) #[B,1]
            NSA_values, next_indices = self.network(next_state_batch) #[B,2], [B,1]
            NSA_values = NSA_values.max(1)[0].unsqueeze(1).to(device) #[B,1]
            fixed_Q_values, _ = self.network(torch.tensor([self.s0]).unsqueeze(1).to(device)) #[1,2], [1,1]
            fixed_Q_values = fixed_Q_values.gather(1, torch.tensor([self.a0],dtype=torch.int64).unsqueeze(1).to(device)) #[1,1]
            targets = (1-action_batch)*(reward_batch+indices)+\
            action_batch*reward_batch + NSA_values - fixed_Q_values #[B,1]
            avg_part = targets-SA_values #[B,1]
            assert avg_part.requires_grad == False, "avg_part requires grad"
        # ================================================================
        
        # ========Calculating the gradient term =================
        state = torch.tensor([trans.state],dtype=torch.float32).unsqueeze(1).to(device)                 #[1,1]
        action = torch.tensor([trans.action],dtype=torch.int64).unsqueeze(1).to(device)                 #[1,1]
        reward = torch.tensor([trans.reward],dtype=torch.float32).unsqueeze(1).to(device)               #[1,1]
        next_state = torch.tensor([trans.next_state],dtype=torch.float32).unsqueeze(1).to(device)       #[1,1]
        SA_value, index = self.network(state)       #[1,2], [1,1]                                          
        SA_value = SA_value.gather(1, action).to(device)    #[1,1]                                                      
        NSA_value, next_index = self.network(next_state)     #[1,2], [1,1]                           
        NSA_value = NSA_value.max(1)[0].unsqueeze(1).to(device)  #[1,1]                                                  
        fixed_Q_value, _ = self.network(torch.tensor([self.s0]).unsqueeze(1).to(device))   #[1,2], [1,1]
        fixed_Q_value = fixed_Q_value.gather(1, torch.tensor([self.a0],dtype=torch.int64).unsqueeze(1).to(device))        #[1,1]
        target = (1-action)*(reward+index)+\
            action*reward + NSA_value - fixed_Q_value #[1,1]
        assert target.requires_grad == True, "target doesn't requires grad"
        assert SA_value.requires_grad == True, "SA_value doesn't requires grad"
        # ================================================================
        avg_part = torch.cat([avg_part,(target-SA_value).detach()]) #[B+1,1]
        # tensor.detach() creates a tensor that shares storage with tensor that does not require grad.
        avg_part = torch.mean(avg_part) #[]
        assert avg_part.requires_grad == False, "avg_part requires grad"
        # ================================================================
        loss = torch.mul(avg_part,target-SA_value) #[1]
        self.optimizer_Q.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_Q.step()

        actual_loss = F.mse_loss(SA_value,target)
        return actual_loss.item()


    def Optimise_whittle(self,Samples,trans):
        batch = self.Transition(*zip(*Samples))
        state_batch = torch.tensor(batch.state,dtype=torch.float32).unsqueeze(1).to(device)                 #[B,1]
        action_batch = torch.tensor(batch.action,dtype=torch.int64).unsqueeze(1).to(device)               #[B,1]
        reward_batch = torch.tensor(batch.reward,dtype=torch.float32).unsqueeze(1).to(device)               #[B,1]
        next_state_batch = torch.tensor(batch.next_state,dtype=torch.float32).unsqueeze(1).to(device)       #[B,1]
        len1 = action_batch.size()[0] #B
        with torch.no_grad():
            SA_values, indices = self.network(state_batch)
            actions1 = torch.tensor([1.]* len1,dtype=torch.int64).unsqueeze(1).to(device) #[B,1]
            Q_values = SA_values.gather(1, actions1).to(device) #[B,1]
            NSA_values, next_indices = self.network(next_state_batch) #[B,1]
            NSA_values = NSA_values.max(1)[0].unsqueeze(1).to(device) #[B,1]
            fixed_Q_values, _ = self.network(torch.tensor([self.s0]).unsqueeze(1).to(device)) #[1,2], [1,1]
            fixed_Q_values = fixed_Q_values.gather(1, torch.tensor([self.a0],dtype=torch.int64).unsqueeze(1).to(device)) #[1,1]
            targets = Q_values-reward_batch+fixed_Q_values-NSA_values #[B]
            avg_part = targets-indices #[B]
            assert avg_part.requires_grad == False, "avg_part requires grad"
        # ================================================================
        
        # ========Calculating the gradient term =================
        state = torch.tensor([trans.state],dtype=torch.float32).unsqueeze(1).to(device)                 #[1,1]
        action = torch.tensor([trans.action],dtype=torch.float32).unsqueeze(1).to(device)               #[1,1]
        reward = torch.tensor([trans.reward],dtype=torch.float32).unsqueeze(1).to(device)               #[1,1]
        next_state = torch.tensor([trans.next_state],dtype=torch.float32).unsqueeze(1).to(device)       #[1,1]
        SA_value, index = self.network(state)                                                #[1,1]
        action1 = torch.tensor([1.],dtype=torch.int64).unsqueeze(1).to(device)                                            #[1,1]
        Q_value = SA_value.gather(1, action1).to(device)                                                           #[1,1]

        NSA_value, next_index = self.network(next_state) #[1,1]
        NSA_value = NSA_value.max(1)[0].unsqueeze(1).to(device) #[B,1]
        fixed_Q_value, _ = self.network(torch.tensor([self.s0]).unsqueeze(1).to(device)) #[1,2], [1,1]
        fixed_Q_value = fixed_Q_value.gather(1, torch.tensor([self.a0],dtype=torch.int64).unsqueeze(1).to(device)) #[1,1]
        target = Q_value-reward+fixed_Q_value-NSA_value #[1]
        assert target.requires_grad == True, "target doesn't requires grad"
        # ================================================================
        avg_part = torch.cat([avg_part,(target-index).detach()]) #[B+1]
        # tensor.detach() creates a tensor that shares storage with tensor that does not require grad.
        avg_part = torch.mean(avg_part) #[]
        assert avg_part.requires_grad == False, "avg_part requires grad"
        # ================================================================
        loss = torch.mul(avg_part,target-index) #[1]
        self.optimizer_whittle.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_whittle.step()

        actual_loss = F.mse_loss(index,target)
        return actual_loss.item()
        

    def get_indices(self):
        x = []
        for i in range(1,self.state_size+1):
            q_value, index = self.network(torch.tensor([i]).unsqueeze(1).to(device))
            x.append(round(index.item(),3))
        return x

    def get_Q(self,state):
        with torch.no_grad():
            action0 = torch.tensor([0.]).unsqueeze(1).to(device)
            action1 = torch.tensor([1.]).unsqueeze(1).to(device)
            Q0 = self.network(state.unsqueeze(1).to(device),action0)
            Q1 = self.network(state.unsqueeze(1).to(device),action1)
            best_action = torch.tensor([0.]) if Q0>=Q1 else torch.tensor([1.])
        assert best_action.requires_grad == False, "something is wrong"
        #print("best action", best_action)
        return best_action.to(torch.device("cpu"))

    def get_Q_values(self):
        with torch.no_grad():
            Q = []
            for s in range(1,self.state_size+1):
                A = self.network(torch.tensor([s],dtype=torch.float32).unsqueeze(1).to(device)).squeeze(0)
                Q.append(A.cpu().detach().numpy())
        return Q