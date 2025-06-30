import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedLanderNet(nn.Module):
    """
    A flexible feedforward neural network for LunarLander.
    You can specify any number of hidden layers and sizes.
    """
    def __init__(self, structure):
        """
        Args:
            structure: list like [8, 32, 16, 4]
                       input_dim=8, output_dim=4, rest are hidden layers
        """
        super().__init__()
        self.structure = structure
        self.layers = nn.ModuleList()
        for i in range(len(structure) - 1):
            self.layers.append(nn.Linear(structure[i], structure[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)  # ReLU for hidden layers only
        return x

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
