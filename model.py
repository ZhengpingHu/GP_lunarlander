import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedLanderNet(nn.Module):
    """
    A feedforward neural network for LunarLander using sliding window and output feedback.
    """

    def __init__(self, window_size=4, state_dim=8, action_dim=4, hidden_dims=[64, 32], output_dim=4):
        """
        Args:
            window_size: how many time steps to stack
            state_dim: dimension of raw state
            action_dim: dimension of action logits used as input feedback
            hidden_dims: list of hidden layer sizes
            output_dim: number of output actions (discrete)
        """
        super().__init__()
        self.window_size = window_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        input_dim = window_size * (state_dim + action_dim)
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.output_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return self.output_layer(x)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def get_flat_params(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat_params):
        i = 0
        for p in self.parameters():
            shape = p.shape
            num = p.numel()
            p.data = flat_params[i:i + num].view(shape).clone()
            i += num
