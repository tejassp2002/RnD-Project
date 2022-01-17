import math
# import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from itertools import count
from collections import namedtuple

## Import function approximator for Q from other file
from QNetwork import QNetwork


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


device = torch.device("cuda")
torch.set_default_dtype(torch.float32)



class DQN_agent:
    """Fixed -size buffe to store experience tuples."""
    def __init__(self, state_size=1, action_size=2, batch_size=3, seed=123):
        """ Initialize a ReplayBuffer object.
        ======
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.seed = np.random.seed(seed)

        self.p = 0.2
        # Initialises the Q- Network
        self.qnetwork_local = QNetwork(state_size, action_size).float().to(device)


        # Define the Optimiser
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=0.001)

        # parameters for avg reward
        #s0 is 0 age of forest
        self.s0 = 0.
        # a0 is wait action
        self.a0 = 1.


    def utility(self,state,action):
        # Geometeric utility
        # Action 0 - wait; 1 - Cut
        if action == 1:
            # reward = current age of the forest
            return state
        else:
            # else no reward
            return torch.tensor([0.])


    def step(self, state, action):
        # extract current state, next state , action and reward
        current_action = action
        '''Current states'''
        current_state = state.clone()
        '''Evaluate immediate rewards and next states'''
        immediate_reward = self.utility(current_state, current_action)

        if current_action == 1.0:
            # if action is to "cut" then the forest age goes to 0
            next_state = torch.tensor([0.])
        else:
            # wait action
            if np.random.uniform() < self.p:
                # with probability p it burns the fraction of the forest
                # the fraction to burn is done uniformly
                temp1 = int(10*current_state) + 1
                #Return random integers from low (inclusive) to high (exclusive).
                next_state = np.random.randint(temp1,size =1)/10
                next_state = torch.tensor(next_state,dtype=torch.float32)
            else:
                # else with probability 1-p we increase the age of the state
                # 0.1 increase corresponds to age increase of 1
                # maximum age is given by state size
                if int(10*current_state) == self.state_size:
                    # if forest reaches it's maximum age it stays there
                    # unless there is a fire or action ‘Cut’ is performed
                    next_state = current_state
                else:
                    next_state = current_state + 0.1

        return (current_state, current_action, immediate_reward, next_state)

    def optimise_model(self,trans):
        # ========Calculating the term under the overline=================
        # average term with fixed state and action pair
        # Transpose the batch. This converts batch-array of Transitions to Transition of batch-arrays.
        # print(Samples)
        # ========Calculating the gradient term =================
        state = torch.tensor([trans.state]).unsqueeze(1).to(device) #[1,1]
        action = torch.tensor([trans.action]).unsqueeze(1).to(device) #[1,1]
        reward = torch.tensor([trans.reward]).to(device) #[1]
        next_state = torch.tensor([trans.next_state]).unsqueeze(1).to(device) #[1,1]
        action0 = torch.tensor([0.]).unsqueeze(1).to(device) #[1,1]
        action1 = torch.tensor([1.]).unsqueeze(1).to(device) #[1,1]
        with torch.no_grad():
            NSA_value0 = self.qnetwork_local(next_state, action0) #[1]
            NSA_value1 = self.qnetwork_local(next_state, action1) #[1]
            NSA_value = torch.max(NSA_value0, NSA_value1) #[1]
            fixed_Q_value = self.qnetwork_local(
                torch.tensor([self.s0]).unsqueeze(1).to(device),
                torch.tensor([self.a0]).unsqueeze(1).to(device)) #[1]
            target = reward+NSA_value - fixed_Q_value #[1]
        assert target.requires_grad == False, "target requires grad"
        SA_value = self.qnetwork_local(state, action) #[1]
        # ================================================================
        loss = F.mse_loss(SA_value,target) #[1]
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()


    def get_Q(self,state):
        with torch.no_grad():
            action0 = torch.tensor([0.]).unsqueeze(1).to(device)
            action1 = torch.tensor([1.]).unsqueeze(1).to(device)
            Q0 = self.qnetwork_local(state.unsqueeze(1).to(device),action0)
            Q1 = self.qnetwork_local(state.unsqueeze(1).to(device),action1)
            best_action = torch.tensor([0.]) if Q0>=Q1 else torch.tensor([1.])
        assert best_action.requires_grad == False, "something is wrong"
        #print("best action", best_action)
        return best_action.to(torch.device("cpu"))
