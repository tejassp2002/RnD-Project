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

def fgdqn_train(init_iters=10,iterations = 10000):
    Q_gradient_steps = 0
    W_gradient_steps = 0
    running_q_loss=0
    running_whittle_loss=0
    # filling the memory with random transitions
    for itr in range(init_iters):
        for state in range(1,state_size+1):
            states = np.full(I,state)
            np.random.shuffle(actions)
            transitions = env.get_transitions(states,actions)
            # transitions are Named Tuples of int
            memory.push_transition(transitions)
    
    for itr in range(iterations-init_iters):
        # ========== filling the memory with random transitions ==========
        for state in range(1,state_size+1):
            states = np.full(I,state)
            np.random.shuffle(actions)
            transitions = env.get_transitions(states,actions)
            # transitions are Named Tuples of int
            memory.push_transition(transitions)
        # ================================================================
        # ========================= Q-gradient step ======================
            
        transitionq = memory.sample_trans_Q() # single transition sampled randomly
        mod_samples = memory.sample_batch(transitionq[0])
        # mod_samples are transitions with fixed state-action pair as of the transitionq
        # excluding the current transition i.e. transitionq[0]
        # takes a single Q gradient step
        loss_q = Agent.Optimise_Q(mod_samples,transitionq[0])
        Q_gradient_steps += 1
        Q_losses.append(loss_q)
        writer.add_scalar("Q Loss", loss_q, Q_gradient_steps)
        running_q_loss += loss_q

        if Q_gradient_steps%running_time_q==0:
            print(f"Running Q Loss {Q_gradient_steps}: {running_q_loss/running_time_q}")
            writer.add_scalar("Running Q Loss", running_q_loss/running_time_q, Q_gradient_steps)
            running_q_loss = 0.0
        # ================================================================
        # ================ Whittle Index -gradient step ==================
        if itr%whittle_every==0:
            transitionw = memory.sample_trans_whittle()  # single transition sampled randomly with action = 0 
            # samples with fixed state and action = 0 as of the transitionw
            w_samples = memory.sample_batch(transitionw[0])
            # w_samples are transitions with fixed state-action pair as of the transitionw 
            # excluding the current transition i.e. transitionw[0]
            # takes a single Whittle Index gradient step
            loss_w = Agent.Optimise_whittle(w_samples,transitionw[0])
            W_gradient_steps += 1
            W_losses.append(loss_w)
            writer.add_scalar("W Loss", loss_w, W_gradient_steps)
            running_whittle_loss += loss_w

            x = Agent.get_indices()
            for i in range(len(x)):
                writer.add_scalar(f"Whittle Index {i+1}",x[i],W_gradient_steps)

            if W_gradient_steps%running_time_whittle==0:
                print(f"Whittle Loss {W_gradient_steps}: {running_whittle_loss/running_time_whittle}| Indices: {Agent.get_indices()}")
                writer.add_scalar("Running Whittle Loss", running_whittle_loss/running_time_whittle, W_gradient_steps)
                running_whittle_loss = 0.0

        # ================================================================

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
    q_lr = 1e-3
    whittle_lr = 1e-3
    Agent = FGDQN_agent(q_lr=q_lr, whittle_lr=whittle_lr, state_size=state_size,seed=args.seed)
    
    #initializing the replay memory
    memory = ReplayMemory(100000)

    #initializing the hyperparameters
    I = 100
    K = 20   #activating K out of I arms at a time
    actions = np.zeros(I) 
    actions[:K] = 1   

    whittle_every = 5          # update whittle network every iterations
    
    #time window to calculate running average of loss    
    running_time_q = 40         
    running_time_whittle = 40
    Q_losses = []
    W_losses = []

    fgdqn_train(10,args.n_iterations)

    print(Agent.get_indices())
