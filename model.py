"""Router and expert networks for prediction-driven MRI subtyping."""

import torch
from torch import nn


class MLP(nn.Module):
    """Three hidden Linear/ReLU/Dropout blocks followed by a linear head.

    This retains the architecture in the original implementation; the paper
    describes it as a three-layer MLP. Outputs are unnormalized logits/risks.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        blocks = []
        for index in range(3):
            blocks.extend([
                nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.network = nn.Sequential(*blocks, nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.network(x)


class SubtypingMoE(nn.Module):
    """Return router logits [N, K] and all expert outputs [N, K]."""

    def __init__(self, input_dim=96, experts=4, hidden_dim=128,
                 expert_dropout=0.5, router_dropout=0.2):
        super().__init__()
        if input_dim < 1 or experts < 2 or hidden_dim < 1:
            raise ValueError("Require positive dimensions and at least two experts.")
        self.router = MLP(input_dim, experts, hidden_dim, router_dropout)
        self.experts = nn.ModuleList([
            MLP(input_dim, 1, hidden_dim, expert_dropout) for _ in range(experts)
        ])

    def forward(self, x):
        return self.router(x), torch.cat([expert(x) for expert in self.experts], dim=1)
