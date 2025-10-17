import torch
import random
import torch.nn as nn
import numpy as np
import time
from torch import Tensor as T
import collections
from torch.utils.data import Dataset
from torch import optim


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


def get_time_str(add_time: bool = True):
    my_str = 'Date_%Y-%m-%d'
    if add_time:
        my_str += ',Time_%H-%M-%S'
    return time.strftime(my_str, time.localtime())


def set_seed(seed):
    """Sets the seed of random number generators to the predefined seed number for reproducibility.
    """
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def freeze_or_unfreeze(obj: nn.Module, requires_grad: bool):
    if obj is None:
        print("WARNING: The object is None. It has no parameter to freeze!")
        return
    # for name, param in obj.named_parameters():
    if isinstance(obj, nn.Parameter):
        obj.requires_grad = requires_grad
    
    for param in obj.parameters():
        param.requires_grad = requires_grad
    
    # obj.train(mode=requires_grad)
    
    
def are_consecutive(numbers_list: list):
    if len(numbers_list) < 2:
        return True
    
    sorted_numbers = sorted(numbers_list)
    
    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i + 1] != sorted_numbers[i] + 1:
            return False
    
    return True


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device, non_blocking=True)
    elif isinstance(input, str) or isinstance(input, int) or isinstance(input, float) or input is None:
        return input
    elif isinstance(input, collections.abc.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.abc.Sequence):
        return [to_device(sample, device=device) for sample in input]
    elif isinstance(input, set):
        return {to_device(itm, device=device) for itm in input}
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")
    
    
def get_object_name(obj):
    for name, value in globals().items():
        if value is obj:
            return name
    
    
class AverageAccuracyCalculator:
    def __init__(self):
        self.count = 0
        self.sum = 0.0

    def update(self, labels_inferenced: T, labels_real: T):
        self.sum += ((labels_inferenced == labels_real).float()).sum().item()
        self.count += len(labels_real) * 1.0

    def calculate(self):
        if self.count > 0:
            return 100.0 * self.sum / self.count
        else:
            return -1.0


def get_params_groups(model: nn.Module, name_model: str, lr=-1, weight_decay=-1, force_considering_as_non_reqularized=False):
    if model is None or lr == 0:
        return []
    regularized = []
    not_regularized = []
    
    if isinstance(model, nn.Parameter):
        if model.requires_grad:
            not_regularized.append(model)
    else:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1 or param.numel() == 1 or force_considering_as_non_reqularized:
                not_regularized.append(param)
            else:
                regularized.append(param)
    
    regularized_dict = {'params': regularized, 'name': name_model + ' regularized'}
    not_regularized_dict = {'params': not_regularized, 'weight_decay': 0., 'name': name_model + ' not_regularized'}
    
    if lr != -1:
        regularized_dict['lr'] = lr
        not_regularized_dict['lr'] = lr
        
    if weight_decay != -1:
        regularized_dict['weight_decay'] = weight_decay
        
    result = []
    
    if len(regularized_dict['params']) > 0:
        result.append(regularized_dict)
    
    if len(not_regularized_dict['params']) > 0:
        result.append(not_regularized_dict)
    
    return result


def show_number_of_parameters_in_pramas_groups(params_all: list, logger):
    num_parameters = 0
    
    for param_list1 in params_all:
        num_parameters_comp = 0
        
        for p in param_list1['params']:
            if p.requires_grad:
                num_parameters_comp += p.numel()
    
        num_parameters += num_parameters_comp
        logger.info(f"Number of learnable parameters of {param_list1['name']}: {num_parameters_comp}")
    
    logger.info(f'Total number of learnable parameters: {num_parameters}')
    
    
class DatasetFromTensors(Dataset):
    def __init__(self, *tensors):
        super().__init__()
        
        self.tensors = tensors
        
        assert len(tensors) > 0
        
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                if i != j and len(tensors[i]) != len(tensors[j]):
                    raise "Error: The tensors have have different number of elements!"
            
        self.num_samples = len(self.tensors[0])
        self.index = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        
        result = []
        
        for t in self.tensors:
            result.append(t[index])
        
        return result
    
    
def get_optimizer_from_params(params_all, optimizer_name: str, lr_default: float, weight_decay: float):
    assert len(params_all) > 0
    
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            params_all,
            momentum=0.9,
            lr=lr_default,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            params_all,
            lr=lr_default,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            params_all,
            lr=lr_default,
            weight_decay=weight_decay
        )
        
    return optimizer


def get_printable_string_from_a_list_of_float_numbers_with_two_digits(numbers_list: list):
    results = [f'{acc:.2f}' for acc in numbers_list]
        
    results = ', '.join(results)
    
    results = '[' + results + ']'
    
    mean_str = f'{np.mean(numbers_list):.2f}'
    
    return results, mean_str
