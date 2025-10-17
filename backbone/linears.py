'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F


class CosineLinearFeature(nn.Module):
    def __init__(self, input_dim, output_dim, sigma=True):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = self.output_dim = output_dim
        self.sigma = sigma
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)
    
    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def forward(self, input, return_dict: bool = True):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.sigma is not None:
            out = self.sigma * out

        if return_dict:
            out = {'logits': out}
        return out
