# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from forest import ForestManagement
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from fgdqn import FGDQN
from dqn import DQN

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Forest_Management",help="the environment id")
    parser.add_argument("--max-age", type=int, default=10,help="the maximum age of the forest")
    parser.add_argument("--total-gradsteps", type=int, default=10000,
        help="total gradient steps to take")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=None,
        help="the replay memory buffer size")
    parser.add_argument("--batch-size", type=int, default=None,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=1000,
        help="timestep to start learning")
    parser.add_argument("--eval-frequency", type=int, default=10,
        help="the frequency of evaluation")
    parser.add_argument("--log-interval", type=int, default=4,help="the interval of logging")
    parser.add_argument("--target-network-frequency", type=int, default=None,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--agent", type=str, default="FGDQN",help="the agent to use")
    parser.add_argument("--avg-reward", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="True if objective is average reward")
    parser.add_argument("--gamma", type=float, default=None,help="the discount factor")
    args = parser.parse_args()
    # fmt: on
    return args




if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print('{} : {}'.format(arg, getattr(args, arg)))
    run_name = f"{args.env_id}__{args.agent}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    os.mkdir(f"runs/{run_name}/videos")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)

    # env setup
    env = ForestManagement(args.max_age,args.seed)

    if args.agent == "FGDQN":
        model = FGDQN(env, device, args)
    elif args.agent == "DQN":
        model = DQN(env, device, args)

    eval_env = ForestManagement(args.max_age,args.seed)
    model.learn(writer,args,eval_env=eval_env)
    
    writer.close()