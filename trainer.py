import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
import time
import pickle
import torchvision
import GPUtil
import our_utils as ou
import platform
from collections import defaultdict
from box import Box
from torch import nn


def train(args: Box):
    seed_list = copy.deepcopy(args.seed)
    device = copy.deepcopy(args.device)

    for seed in seed_list:
        args.seed = seed
        args.device = device
        _train(args)


def _train(args: Box):

    logs_directory_name = args.logdir
    
    if not os.path.exists(logs_directory_name):
        os.makedirs(logs_directory_name)

    log_file_name = "{}/DB={},{},Order={},Seed={},Time{}".format(
        logs_directory_name,
        args.dataset,
        args.prefix,
        args.order,
        args.seed,
        ou.get_time_str()
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s:%(lineno)d] => %(message)s",
        handlers=[
            logging.FileHandler(filename=log_file_name + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    ou.set_seed(args.seed)
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args.dataset,
        args.shuffle,
        args.seed,
        args.init_cls,
        args.increment,
        args,
    )
    
    args.class_order = data_manager._class_order
    
    model = factory.get_model(model_name=args.model_name, args=args)
    
    last_finished_task_id = -1
    if args.load_path != '' and args.load_path is not None:
        last_finished_task_id = load_model(model, args.load_path)

    DIL_accuracies_without_oracle_list = []
    DIL_accuracies_with_oracle_list = []
    domain_classification_accuracy_list = []
    
    accuracies_dict_to_save = defaultdict(dict)
    
    for domain_id in range(last_finished_task_id + 1, data_manager.nb_tasks):
        args.current_task_id = domain_id
        
        model.incremental_train(data_manager)
        
        if not args.UMAP:
            accuracies_without_oracle_dict, accuracies_with_oracle_dict, domain_classification_accuracy = model.eval_task()
            
            accuracies_dict_to_save[domain_id] = {
                'accuracies_without_oracle_dict': accuracies_without_oracle_dict,
                'accuracies_with_oracle_dict': accuracies_with_oracle_dict,
                'domain_classification_accuracy': domain_classification_accuracy
            }
        
        model.after_task()
        
        if args.UMAP:
            continue
        
        logging.info("CNN (w/o oracle): {}".format(accuracies_without_oracle_dict["grouped"]))
        
        DIL_accuracies_without_oracle_list.append(accuracies_without_oracle_dict['top1'])
        
        temp, mean_str = ou.get_printable_string_from_a_list_of_float_numbers_with_two_digits(DIL_accuracies_without_oracle_list)
        logging.info(f'Accuracies (w/o oracle) for order {args.order}: {temp} -> Avg.: {mean_str}')
        
        if args.dataset != 'core50':   # For CORe50, the test set does not contain the domain information.
            logging.info("CNN (with oracle): {}".format(accuracies_with_oracle_dict["grouped"]))
            
            DIL_accuracies_with_oracle_list.append(accuracies_with_oracle_dict['top1'])
            
            temp, mean_str = ou.get_printable_string_from_a_list_of_float_numbers_with_two_digits(DIL_accuracies_with_oracle_list)
            logging.info(f'Accuracies (with oracle) for order {args.order}: {temp} -> Avg.: {mean_str}')
            
            domain_classification_accuracy_list.append(domain_classification_accuracy)
            temp, mean_str = ou.get_printable_string_from_a_list_of_float_numbers_with_two_digits(domain_classification_accuracy_list)
            logging.info(f'Domain classification accuracies (after each domain) for order {args.order}:  {temp} -> Avg.: {mean_str}')
            
        if args.save_model_after_each_task:
            save_model(model=model, args=args, file_path=log_file_name + f",model,after_task_{domain_id}.pth")
    
    with open(log_file_name + ".pkl", "wb") as f:
        pickle.dump(accuracies_dict_to_save, f)
    
    logging.info('The process is finished!')
    
    if args.save_last_model:
        save_model(model=model, args=args, file_path=log_file_name + ",final_model.pth")
        
    # To draw the UMAP
    return model


def save_model(model: nn.Module, args: Box, file_path: str):
    # We save args to know what was the setting in the past.
    state = args.to_dict()
    state['model_state'] = model._network.state_dict()
    torch.save(state, file_path)
    logging.info(f'The parameters are saved after task: {args.current_task_id}!')
    

def load_model(model: nn.Module, file_path: str):
    args_dict = torch.load(file_path)
    args = Box(args_dict)
    
    model._network.load_state_dict(args.model_state)
    current_task_id = args.current_task_id
    
    args.pop('model_state')
    
    logging.info(f'The parameters are loaded for task: {args.current_task_id}!')
    return current_task_id


def _set_device(args):
    device_type = args.device
    gpu_list = []

    if type(device_type) in [int, str]:
        device_type = [f'{device_type}']
    
    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpu_list.append(device)

    args.device = gpu_list


def print_args(args):
    
    separator = '-' * 100
    logging.info(separator)
    gpus = GPUtil.getGPUs()
    current_gpu = gpus[args.device[0].index]
    # logging.info(f"ID: {current_gpu.id}")
    logging.info(f'Python version: {platform.python_version()}')
    logging.info(f'PyTorch version: {torch.__version__},TorchVision version: {torchvision.__version__}')
    logging.info(f"GPU Name: {current_gpu.name}")
    logging.info(f"Driver: {current_gpu.driver}")
    
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
    logging.info(separator)
