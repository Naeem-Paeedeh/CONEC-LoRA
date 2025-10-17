import torch.nn as nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 dim_input: int,
                 dim_hidden: int,
                 dim_output: int,
                 num_layers: int,
                 bias: bool = True,
                 dropout_rate: float = 0.0,
                 device=None,
                 dtype=None,
                 use_BatchNorm=False
                 ):
        
        super().__init__()

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.bias = bias
        self.device = device
        self.num_layers = num_layers
        self.use_BatchNorm = use_BatchNorm
        
        assert self.num_layers >= 1
        
        self.layers = nn.ModuleList()
        
        def dim_inp(idx_layer):
            if idx_layer == 0:
                return self.dim_input
            return self.dim_hidden
        
        def dim_out(idx_layer):
            if idx_layer == self.num_layers - 1:
                return self.dim_output
            return self.dim_hidden
            
        if use_BatchNorm:
            self.layers.append(nn.BatchNorm1d(self.dim_input, dtype=dtype, device=device))
        
        self.layers.append(nn.Linear(self.dim_input, dim_out(0), bias=self.bias, device=self.device, dtype=self.dtype))
        
        for i in range(1, self.num_layers):
            if use_BatchNorm:
                self.layers.append(nn.BatchNorm1d(dim_inp(i), dtype=dtype, device=device))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.dropout_rate))
            
            self.layers.append(nn.Linear(dim_inp(i), dim_out(i), bias=self.bias, device=self.device, dtype=self.dtype))

    def forward(self, inp, use_residual_connection=False):
        # inputs' shape is (batch_size, dim_embed)
        # x.shape = (batch_size, 2 * dim_embed)
        x = inp
        
        for lyr in self.layers:
            x = lyr.forward(x)

        if use_residual_connection:
            return x + inp
        
        return x
