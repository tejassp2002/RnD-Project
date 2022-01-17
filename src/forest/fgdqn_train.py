import matplotlib.pyplot as plt
import numpy as np
#from DQN_main_Inbuilt import DQN_agent
from fgdqn_avg import FGDQN_agent
import torch
import csv
import math
from ReplayMemory import ReplayMemory
import argparse


def dqn_train(iterations = 300):
    # initial observation
    state = torch.zeros(1, dtype=torch.float32)
    gradient_steps = 0
    for itr in range(iterations):
        # =============================================================================
        # run a simulation step using epsilon-greedy policy
        # exponential decay of epsilon (exploration parameter)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * itr / EPS_DECAY)
        print("epsilon",eps_threshold)
        # epsilon greedy
        if np.random.rand()<eps_threshold:
            # with probability epsilon select random action
            action = torch.tensor([float(np.random.randint(2))])
        else:
            # with probability 1-epsilon select action which gives maximum Q value
            action = agent.get_Q(state)
            state = state.to(torch.device("cpu"))
        print("action chosen",action)
        (current_state, current_action, immediate_reward, next_state) = agent.step(state,action)
        memory.push_transition(current_state, current_action, immediate_reward, next_state)
        # =============================================================================

        if len(memory) < batch_size:
            print(f"memory not filled yet at iteration {itr} and memory size is {len(memory)}")
            continue
        else:
            batch_sample = memory.sample_batch_train()

        loss_per_batch = 0
        for k in range(batch_size):
            mod_samples = memory.sample_batch_FG(batch_sample[k])
            # mod_samples are transitions with fixed state-action pair
            # excluding the current transition 
            # takes a single gradient step
            loss_per_batch += agent.optimise_model(batch_sample[k],mod_samples)
            gradient_steps += 1
        loss = loss_per_batch/batch_size

        print(f"loss at iteration {itr} is {loss} and memory size is {len(memory)}")
        losses.append(loss)
        # policy at each iteration 
        policy = []
        for j in range(10):
            action = agent.get_Q(torch.tensor([j/10]).to(device))
            policy.append(int(action.item()))
        policy_iters.append(policy)
        Hamming_distance.append(sum(a != b for a, b in zip(policy, optimal_policy)))

        state_record.append(state.item())

        # change the state to the next state
        state = next_state
    
    for j in range(10):
        action = agent.get_Q(torch.tensor([j/10]).to(device))
        final_policy.append(int(action.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter",type=int)
    parser.add_argument("--expname",type=str)
    args = parser.parse_args()

    batch_size = 64 # minibatch size

    EPS_START = 0.99
    EPS_END = 0.05
    EPS_DECAY = 600
    state_size = 10
    action_size = 2
    seed = 12498
    device = torch.device("cuda")
    torch.set_default_dtype(torch.float32)

    replay_memory = 100000

    losses = []
    np.random.seed(seed)

    agent = FGDQN_agent(state_size,action_size,batch_size,seed)
    memory = ReplayMemory(replay_memory,batch_size)
    policy_iters = []
    state_record = []
    Hamming_distance = []
    optimal_policy = [0,0,0,1,1,1,1,1,1,1]
    final_policy = []
    dqn_train(args.iter)

    with open(f'Data/FGDQN_{args.expname}.csv', "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Loss","State","Policy","Hamming_distance"])
        writer.writerows(zip(losses, state_record,policy_iters,Hamming_distance))
        writer.writerow(["Final Policy",final_policy])
    
    print("Training Completed")

