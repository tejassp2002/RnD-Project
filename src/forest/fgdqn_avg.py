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



class FGDQN_agent:
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

    def optimise_model(self,trans,Samples):
        # ========Calculating the term under the overline=================
        # average term with fixed state and action pair
        # Transpose the batch. This converts batch-array of Transitions to Transition of batch-arrays.
        # print(Samples)
        avg_part = 0
        if len(Samples) == 0:
            pass
        else:
            batch = self.Transition(*zip(*Samples))
            state_batch = torch.stack(batch.state,0).to(device) #[B,1]
            action_batch = torch.stack(batch.action,0).to(device) #[B,1]
            reward_batch = torch.stack(batch.reward,0).squeeze(1).to(device) #[B]
            next_state_batch = torch.stack(batch.next_state,0).to(device) #[B,1]
            len1 = action_batch.size()[0]
            with torch.no_grad():
                actions0 = torch.tensor([0.]* len1).unsqueeze(1).to(device) #[B,1]
                actions1 = torch.tensor([1.]* len1).unsqueeze(1).to(device) #[B,1]
                NSA_values0 = self.qnetwork_local(next_state_batch, actions0) #[B]
                NSA_values1 = self.qnetwork_local(next_state_batch, actions1) #[B]
                NSA_values = torch.max(NSA_values0, NSA_values1) #[B]
                SA_values = self.qnetwork_local(state_batch, action_batch) #[B]
                fixed_Q_value = self.qnetwork_local(
                    torch.tensor([self.s0]).unsqueeze(1).to(device),
                    torch.tensor([self.a0]).unsqueeze(1).to(device)) #[1]
                targets = reward_batch +  NSA_values - fixed_Q_value #[B]
                avg_part = targets-SA_values #[B]
                assert avg_part.requires_grad == False, "avg_part requires grad"

        # ================================================================
        
        # ========Calculating the gradient term =================
        state = torch.tensor([trans.state]).unsqueeze(1).to(device) #[1,1]
        action = torch.tensor([trans.action]).unsqueeze(1).to(device) #[1,1]
        reward = torch.tensor([trans.reward]).to(device) #[1]
        next_state = torch.tensor([trans.next_state]).unsqueeze(1).to(device) #[1,1]
        action0 = torch.tensor([0.]).unsqueeze(1).to(device) #[1,1]
        action1 = torch.tensor([1.]).unsqueeze(1).to(device) #[1,1]
        NSA_value0 = self.qnetwork_local(next_state, action0) #[1]
        NSA_value1 = self.qnetwork_local(next_state, action1) #[1]
        NSA_value = torch.max(NSA_value0, NSA_value1) #[1]
        SA_value = self.qnetwork_local(state, action) #[1]
        fixed_Q_value = self.qnetwork_local(
            torch.tensor([self.s0]).unsqueeze(1).to(device),
            torch.tensor([self.a0]).unsqueeze(1).to(device)) #[1]
        target = reward+NSA_value - fixed_Q_value #[1]
        assert target.requires_grad == True, "target doesn't requires grad"
        avg_part = avg_part+(target-SA_value).detach() #[B+1]
        # tensor.detach() creates a tensor that shares storage with tensor that does not require grad.
        avg_part = torch.mean(avg_part) #[]
        assert avg_part.requires_grad == False, "avg_part requires grad"
        # ================================================================
        loss = torch.mul(avg_part,target-SA_value) #[1]
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        actual_loss = F.mse_loss(SA_value,target)
        return actual_loss.item()


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
