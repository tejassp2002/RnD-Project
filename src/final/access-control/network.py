import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_states, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions),
        )

    def forward(self, x):
        return self.network(x)