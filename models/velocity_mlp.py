import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return torch.sigmoid(x) * x

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim=cfg.input_dim
        self.hidden_num=cfg.hidden_num
        self.input_layer = nn.Linear(self.input_dim + 1, self.hidden_num, bias=False)
        self.output_layer = nn.Linear(self.hidden_num, self.input_dim, bias=False)
        self.act = Swish()

    def forward(self, x_input, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        elif t.dim() == 0:
            t = t.view(1, 1).expand(x_input.shape[0], 1)
        inputs = torch.cat([x_input, t], dim=1)  
        x = self.input_layer(inputs)
        x = self.act(x)
        x = self.output_layer(x)
        return x