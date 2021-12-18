from fgdqn import FGDQN
import gym
import gym_ple
import highway_env
import numpy as np
import argparse
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure, read_csv

def train(args):
    env = gym.make(args.env)
    base_path = f"./{args.env}/{args.run}"

    # Create the evaluation environment and callbacks  
    eval_env = Monitor(gym.make(args.env))
    eval_callback = EvalCallback(eval_env, best_model_save_path=base_path+"/best_model",
                             log_path=base_path+"/eval_logs", eval_freq=args.eval_freq,
                             deterministic=True, render=False)

    if args.save_freq is None:
        save_freq = args.n_timesteps
    else:
        save_freq = args.save_freq
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=base_path+"/checkpoints",
                                         name_prefix='fgdqn')

    callback = CallbackList([checkpoint_callback, eval_callback])

    # set up logger
    new_logger = configure(base_path+"/logs", ["log", "tensorboard", "csv","stdout"])

    hyperparams = dict(
            policy=args.policy,
            env=args.env,
            policy_kwargs=dict(net_arch=[256,256]),
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.exploration_final_eps,
            verbose=1,
            tensorboard_log=base_path+"/tb", 
            seed=args.seed
        )

    model = FGDQN(**hyperparams)   
    model.set_logger(new_logger)
    
    print("||Starting Training||")
    model.learn(int(args.n_timesteps), callback=callback)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Catcher-v0", help="Environment to train on")
    parser.add_argument("--run", type=int, default=0, help="Run number")
    parser.add_argument("--seed", type=int, default=0,  help="Random seed")
    parser.add_argument("--n_timesteps", type=int, default=2e4, help="Number of timesteps to train for")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--exploration_fraction", type=float, default=0.7, help="Fraction of entire training period over which the exploration rate is annealed")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05, help="Final value of random action probability")
    parser.add_argument("--save_freq", type=int, default=None, help="Save model every n steps")
    parser.add_argument("--train_freq", type=int, default=1, help="Train model every n steps")
    parser.add_argument("--eval_freq", type=int, default=500, help="Evaluate model every n steps")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient steps to take per update")
    parser.add_argument("--learning_starts", type=int, default=100, help="How many steps of the model to collect transitions for before learning starts")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Size of the replay buffer")
    parser.add_argument("--policy", type=str, default="MlpPolicy", help="Policy class to use")

    args = parser.parse_args()
    
    train(args)

    
    
    

    
    