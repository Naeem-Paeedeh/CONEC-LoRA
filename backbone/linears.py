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

    # For CL-LoRA
    def forward_diagonal(
        self,
        input,
        cur_task: int,
        init_cls: int = 10,
        inc: int = 10,
        out_dim: int = 768,
        use_init_ptm: bool = False,
        return_dict: bool = True
    ):
        out_all = None

        for i in range(cur_task + 1):
            if i == 0:
                start_cls, end_cls = 0, init_cls
            else:
                start_cls = init_cls + (i - 1) * inc
                end_cls = start_cls + inc

            # use_init_ptm prepends one extra feature block (the raw ViT features).
            block_index = i + 1 if use_init_ptm else i

            input_i = F.normalize(input[:, block_index * out_dim:(block_index + 1) * out_dim], p=2, dim=1)
            weight_i = F.normalize(self.weight[start_cls:end_cls, block_index * out_dim:(block_index + 1) * out_dim],
                                   p=2, dim=1)

            out = F.linear(input_i, weight_i)
            out_all = out if out_all is None else torch.cat((out_all, out), dim=1)

        if self.sigma is not None:
            out_all = self.sigma * out_all

        if return_dict:
            out_all = {'logits': out_all}
        return out_all
