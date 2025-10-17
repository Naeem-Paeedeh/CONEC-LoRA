import math
import torch
import torch.nn as nn
from torch import Tensor as T
from box import Box


class LoRA_Adapter(nn.Module):
    def __init__(self,
                 args: Box,
                 dim_embed: int,
                 downsize_dimension: int,
                 random_orth=True,
                 ):
        
        super().__init__()
        
        self.random_orth = random_orth

        self.dim_embed = args.d_model if dim_embed is None else dim_embed
        self.down_size = downsize_dimension

        self.B = nn.Linear(self.dim_embed, self.down_size, bias=False)
        self.A = nn.Linear(self.down_size, self.dim_embed, bias=False)

        if self.random_orth:
            random_matrix = torch.rand(self.dim_embed, self.down_size)
            q, r = torch.linalg.qr(random_matrix)
            
            with torch.no_grad():
                self.B.weight.copy_(q.T)
                
            scaling_factor = 1.  # You can adjust this value if needed
            self.B.weight.data *= scaling_factor
        else:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.B.weight, a=math.sqrt(5))

        with torch.no_grad():
            nn.init.zeros_(self.A.weight)

    def forward(self, x: T):
        inter_x = self.B.forward(x)
        out = self.A.forward(inter_x)
        return out
