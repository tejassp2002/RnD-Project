import matplotlib.pyplot as plt
import numpy as np
from FGDQN_Agent import FGDQN_agent
import torch
import csv
import math
from ReplayMemory import ReplayMemory
import argparse
from Environment1 import circulant_dynamics1
from Environment2 import circulant_dynamics2
from collections import namedtuple
import math
from torch.utils.tensorboard import SummaryWriter
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

def fgdqn_train(iterations = 10000):
  running_q_loss=0
  running_whittle_loss=0
  for itr in range(iterations):
    for state in range(1,state_size+1):
        states = np.full(I,state)
        np.random.shuffle(actions)
        transitions = env.get_transitions(states,actions)
        # transitions are Named Tuples of int
        memory.Push_transition(transitions)

        loss_q = 0
        for action in range(2):
            mod_samples = memory.Sample_batch_FG(state,action)
            if mod_samples is not None:
                loss_q += Agent.Optimise_Q(mod_samples)

    loss_whittle = 0
    if itr % whittle_every== 0:
        for i in range(whittle_steps):
            w_samples = memory.Sample_batch_whittle()
            if w_samples is not None:
                loss_whittle += Agent.Optimise_whittle(w_samples)

    running_q_loss += loss_q/(4*2)
    if itr%running_time_q==0:
        #print(f"Q Loss {itr}: {running_q_loss/running_time_q}| Q values: {np.vstack(Agent.get_Q_values())}")
        print(f"Q Loss {itr}: {running_q_loss/running_time_q}")
        writer.add_scalar("Q Loss", running_q_loss/running_time_q, itr)
        running_q_loss = 0.0

    running_whittle_loss += loss_whittle/whittle_steps
    if itr%running_time_whittle==0:
        print(f"Whittle Loss {itr}: {running_whittle_loss/running_time_whittle}| indices: {Agent.get_indices()}")
        writer.add_scalar("Whittle Loss", running_whittle_loss/running_time_whittle, itr)
        running_whittle_loss = 0.0

    x = Agent.get_indices()
    for i in range(len(x)):
        writer.add_scalar(f"Whittle Index {i+1}",x[i],itr)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='FGDQN')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--env', type=int, default=2, help='Environment ID')
    parser.add_argument('--n_iterations', type=int, default=1000, help='number of iterations')
    parser.add_argument('--log_dir', type=str, default="./tb", help='log directory')

    args = parser.parse_args()
    seed = np.random.seed(args.seed)

    writer = SummaryWriter(log_dir=args.log_dir)

    if args.env == 1:
        state_size = 4
        env = circulant_dynamics1()
    elif args.env == 2:
        state_size = 5
        env = circulant_dynamics2()

    #initializing the agent
    q_lr = 1e-4
    whittle_lr = 1e-3
    Agent = FGDQN_agent(q_lr=q_lr, whittle_lr=whittle_lr, state_size=state_size,seed=args.seed)
    
    #initializing the replay memory
    Batch_size = 64
    memory = ReplayMemory(100000,Batch_size)

    #initializing the hyperparameters
    I = 100
    K = 20   #activating K out of I arms at a time
    actions = np.zeros(I) 
    actions[:K] = 1   

    whittle_every = 4           #update whittle network every 4 iterations
    whittle_steps = 4           #gradient steps in one iteration
    
    #time window to calculate running average of loss    
    running_time_q = 40         
    running_time_whittle = 40

    fgdqn_train(args.n_iterations)

    print(Agent.get_indices())
