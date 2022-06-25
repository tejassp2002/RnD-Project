import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, obs_space, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(obs_space).prod(), 512),
            nn.ReLU(),
            # nn.Linear(120, 84),
            # nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.network(x)