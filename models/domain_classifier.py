import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch import Tensor as T
from torch.nn import functional as F

import torch
import torch.nn as nn
from new_types import GenerationStrategy
from box import Box
from collections import defaultdict


class DomainClassifier(nn.Module):
    def __init__(self, args: Box):
        super(DomainClassifier, self).__init__()
        
        self.args = args
        
        self.dim_input = args.embd_dim
        
        if self.args.generation_strategy in [GenerationStrategy.GMM, GenerationStrategy.GMM_with_generator_loss]:
            self.dim_output = args.total_sessions
        else:
            raise NotImplementedError
        
        bias = args.bias_domain_classifier
        
        self.chosen_layers_for_intermediate_domain_classifiers = args.chosen_layers_for_intermediate_domain_classifiers
        
        self.classifiers_dict = nn.ModuleDict()
        
        for block_id in self.chosen_layers_for_intermediate_domain_classifiers:
            self.classifiers_dict[str(block_id)] = nn.Linear(self.dim_input, self.dim_output, bias=bias)
    
    def forward(self, embeddings: T, block_id: int):
        logits = self.classifiers_dict[str(block_id)].forward(embeddings)
        
        return logits
    
    def forward_dictionaries(self, embeddings_dict: dict[T]):
        logits_dict = {}
        
        for block_id in self.chosen_layers_for_intermediate_domain_classifiers:
            block_id_str = str(block_id)
            logits_dict[block_id_str] = self.classifiers_dict[block_id_str].forward(embeddings_dict[block_id_str])
            
        return logits_dict
    

class DomainClassifierData:
    def __init__(self, device):
        self.device = device
        # We keep the lists for chosen layers.
        # For every domain
        
        def init_empty_tensor_float32() -> T:
            return torch.Tensor().to(device=self.device, dtype=torch.float32)
        
        self.GMM_params_dict_of_lists = defaultdict(list)
        # Centers for each domain for the chosen layer.
        self.centers = defaultdict(init_empty_tensor_float32)
        
        self.reset()
        
    def reset(self):
        pass

