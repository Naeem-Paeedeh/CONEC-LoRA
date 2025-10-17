import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, temperature: float = 16):
        super().__init__()
        
        self.input_dim = input_dim
        self.out_dim = self.output_dim = output_dim
        
        self.shape = output_dim, input_dim
        self.mu = nn.Parameter(0.01 * torch.randn(*self.shape, requires_grad=True))
        self.sigma = nn.Parameter(torch.zeros(*self.shape, requires_grad=True))   # each rotation have individual variance here
        self.num_features = input_dim
        self.temperature = temperature

    def forward(self, x, stochastic=True, return_dict: bool = True):
        mu = self.mu
        sigma = self.sigma

        if stochastic:
            sigma = F.softplus(sigma - 4)                                   # when sigma=0, softplus(sigma-4)=0.0181
            weight = sigma * torch.randn_like(mu) + mu
        else:
            weight = mu
        
        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        result = F.linear(x, weight)
        result = result * self.temperature

        if return_dict:
            result = {'logits': result}
        
        return result
    

class Linear2(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = self.output_dim = output_dim
        self.bias = bias
        self.layer = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x, return_dict: bool = True):
        out = self.layer.forward(x)

        if return_dict:
            out = {'logits': out}
        
        return out
