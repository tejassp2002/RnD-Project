import time
from utils import linear_schedule
import random
import torch 
import torch.nn as nn
import numpy as np
from network import QNetwork
from torch.nn import functional as F
import torch.optim as optim
from buffer import FGReplayBuffer
import wandb
import cv2

class DQN(object):
    def __init__(self,env,device,args) -> None:
        self.env = env
        self.actions = env.actions
        self.device = device
        self.q_net = QNetwork(env.num_states,env.num_actions).to(device)
        self.target_net = QNetwork(env.num_states,env.num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.buffer = FGReplayBuffer(
            env.num_states,
            args.buffer_size,
            device,
        )
        self.gradient_steps = 0
        self.avg_reward = args.avg_reward
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.learning_rate)
        if args.avg_reward:
            self.fixed_state = torch.tensor(np.array([env.env_start()]),dtype=torch.float32).to(self.device)
            self.fixed_action = torch.tensor([1],dtype=torch.float32).unsqueeze(0).to(self.device) #[1,1] idle
        else:
            self.gamma = args.gamma

    def train(self, replay_data):
        with torch.no_grad():
            # get the target q values
            target_max, _ = self.target_net(replay_data.next_observations).max(dim=1)
            if self.avg_reward:
                fixed_q_value = self.target_net(self.fixed_state)      
                fixed_q_value = torch.gather(fixed_q_value, dim=1, index=self.fixed_action.long())
                target = replay_data.rewards + target_max.unsqueeze(1) - fixed_q_value
            else:
                target = replay_data.rewards + self.gamma * target_max.unsqueeze(1)

        current_q_value = self.q_net(replay_data.observations)
        current_q_value = torch.gather(current_q_value, dim=1, index=replay_data.actions.long())
        loss = F.mse_loss(current_q_value, target)

        # Optimize the policy
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.gradient_steps += 1
        if self.avg_reward:
            qvalue = fixed_q_value.mean().item()
        else:
            qvalue = current_q_value.mean().item()
        return loss.item(), qvalue


    def learn(self,
        writer,
        args,
        eval_env = None
        ):

        obs = self.env.env_start()
        total_timesteps = args.learning_starts+args.total_gradsteps
        for global_step in range(total_timesteps):
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction *total_timesteps, global_step)
            # perform eps-greedy to get action
            if random.random() < epsilon:
                action = np.random.choice(self.actions)
            else:
                logits = self.q_net(torch.Tensor(obs).unsqueeze(0).to(self.device))
                action = torch.argmax(logits, dim=1).cpu().numpy()

            reward_obs_term = self.env.env_step(action)
            next_obs = reward_obs_term[1]
            reward = reward_obs_term[0]

            self.buffer.add(obs, next_obs, action, reward)
            obs = next_obs

            if global_step > args.learning_starts:
                # take a single gradient step
                # sample data and do the update
                data = self.buffer.sample(args.batch_size)
                loss, qvalue = self.train(data)

                if global_step % args.target_network_frequency == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                # logging
                if global_step % args.log_interval == 0:
                    if args.track:
                        wandb.log({"train/loss": loss, "train/qvalue": qvalue,"gradient_steps": self.gradient_steps})
                    writer.add_scalar("train/loss", loss, self.gradient_steps)
                    writer.add_scalar("train/qvalue", qvalue, self.gradient_steps)

                if self.gradient_steps % args.eval_frequency == 0:
                    # evaluating the agent
                    eval_obs = eval_env.env_start()
                    eval_reward = 0
                    for _ in range(1000):                           
                        eval_logits = self.q_net(torch.Tensor(eval_obs).unsqueeze(0).to(self.device))
                        eval_action = torch.argmax(eval_logits, dim=1).cpu().numpy()
                        eval_terms = eval_env.env_step(eval_action)
                        eval_next_obs = eval_terms[1]
                        eval_reward += eval_terms[0]
                        eval_obs = eval_next_obs

                    if args.track:
                        wandb.log({"eval/reward":eval_reward,\
                            "eval_iter":self.gradient_steps//args.eval_frequency})
                        writer.add_scalar("eval/reward", eval_reward, self.gradient_steps)
                    print("Eval reward:", eval_reward)    

