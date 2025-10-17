import json
import argparse
from trainer import train
from new_types import (ClassifierType, GenerationStrategy)  # They must be detected automatically. Do not remove them!
from our_utils import are_consecutive
from box import Box
from collections.abc import Iterable


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    
    args = Box(args)
    
    args.update(param)  # Add parameters from json
    
    extract_enums(args)
    
    verify_setting(args)
    
    check_required_arguments(args)
    
    set_default_values(args)
    
    train(args)


def verify_setting(args: Box):
    assert args.init_cls == args.increment
    
    if not (are_consecutive(args.LoRA_shared_layers_ids_list) and are_consecutive(args.LoRA_domain_specfic_layers_ids_list)):
        raise NotImplementedError
    
    args.LoRA_shared_layers_ids_list = sorted(args.LoRA_shared_layers_ids_list)
    args.LoRA_domain_specfic_layers_ids_list = sorted(args.LoRA_domain_specfic_layers_ids_list)
    

def check_required_arguments(args: Box):
    def verify_keys(args: Box, required_arguments: Iterable):
        for key in required_arguments:
            if key not in args:
                raise Exception(f'Error: You must set the "{key}" in the "{args.config}" file.')
    
    required_arguments = ['order', 'LoRA_shared_layers_ids_list', 'LoRA_domain_specfic_layers_ids_list', 'lr_default', 'epochs', 'classifier_type']
    
    verify_keys(args=args, required_arguments=required_arguments)
    
    required_arguments = ['margin', 'epochs_domain_classifier_training']
    verify_keys(args=args, required_arguments=required_arguments)
        

def set_default_values(args: dict):
    
    def set_if_it_does_not_exist(args: dict, key: str, default_value):
        if key not in args:
            args[key] = default_value
            
    set_if_it_does_not_exist(args=args, key='logdir', default_value='logs')
    set_if_it_does_not_exist(args=args, key='save_last_model', default_value=False)
    set_if_it_does_not_exist(args=args, key='save_model_after_each_task', default_value=False)
    set_if_it_does_not_exist(args=args, key='load_path', default_value='')
    
    set_if_it_does_not_exist(args=args, key='order', default_value=1)        # Order of the task
    set_if_it_does_not_exist(args=args, key='weight_decay', default_value=2e-4)
    set_if_it_does_not_exist(args=args, key='weight_decay_classifiers', default_value=2e-4)
    set_if_it_does_not_exist(args=args, key='weight_decay_domain_classifiers', default_value=2e-4)
    set_if_it_does_not_exist(args=args, key='min_lr', default_value=1e-8)
    set_if_it_does_not_exist(args=args, key='n_components', default_value=2)
    # set_if_it_does_not_exist(args=args, key='alpha', default_value=0.0)
    # set_if_it_does_not_exist(args=args, key='beta', default_value=0.0)
    
    set_if_it_does_not_exist(args=args, key='classifier_type', default_value=ClassifierType.Separate_CosineLinearLayers)
    lr_default = args.lr_default
    set_if_it_does_not_exist(args=args, key='lr_LoRAs', default_value=lr_default)
    set_if_it_does_not_exist(args=args, key='lr_temporary_classifier', default_value=lr_default)
    set_if_it_does_not_exist(args=args, key='lr_classifier', default_value=lr_default)
    set_if_it_does_not_exist(args=args, key='lr_transformation_module', default_value=1e-4)
    set_if_it_does_not_exist(args=args, key='use_transformation_module', default_value=True)
    set_if_it_does_not_exist(args=args, key='lr_domain_classifiers', default_value=0.02)
    set_if_it_does_not_exist(args=args, key='epochs_domain_classifier_training', default_value=40)
    
    set_if_it_does_not_exist(args=args, key='generation_strategy', default_value=GenerationStrategy.GMM_with_generator_loss)
    set_if_it_does_not_exist(args=args, key='bias_domain_classifier', default_value=True)
    set_if_it_does_not_exist(args=args, key='margin', default_value=1.0)
    set_if_it_does_not_exist(args=args, key='lambda_1', default_value=1.0)
    set_if_it_does_not_exist(args=args, key='lambda_2', default_value=1.0)
    set_if_it_does_not_exist(args=args, key='debugging', default_value=False)
    set_if_it_does_not_exist(args=args, key='chosen_layers_for_intermediate_domain_classifiers', default_value=['final'])
    set_if_it_does_not_exist(args=args, key='confidence_threshold', default_value=[0.5])
    set_if_it_does_not_exist(args=args, key='n_components', default_value=2)
    set_if_it_does_not_exist(args=args, key='LoRA_qkv_mask', default_value=[1, 0, 1])
    set_if_it_does_not_exist(args=args, key='LoRA_shared_layers_ids_list', default_value=[0, 1, 2, 3, 4, 5])
    set_if_it_does_not_exist(args=args, key='LoRA_domain_specfic_layers_ids_list', default_value=[6, 7, 8, 9, 10, 11])
    set_if_it_does_not_exist(args=args, key='LoRA_downsize_dimension', default_value=8)
    set_if_it_does_not_exist(args=args, key='freeze_B_matrices_in_shared_LoRAs', default_value=True)
    set_if_it_does_not_exist(args=args, key='use_proxy_classifier', default_value=True)
    set_if_it_does_not_exist(args=args, key='temperature_stochastic_classifier', default_value=16.0)
    set_if_it_does_not_exist(args=args, key='max_iter_for_GMM', default_value=100)
    set_if_it_does_not_exist(args=args, key='tol_for_GMM', default_value=1e-3)
    
    # When a dataset requires too much memory, we calculate the statistics for a subset of samples.
    set_if_it_does_not_exist(args=args, key='cache_synthetic_embeddings', default_value=True)
    set_if_it_does_not_exist(args=args, key='max_number_of_embeddings_in_memory', default_value=1e8)
    set_if_it_does_not_exist(args=args, key='UMAP', default_value=False)
    
    # We convert all elements to strings to use them as the keys in a dictionary.
    args.LoRA_shared_layers_ids_list = [str(i) for i in sorted(args.LoRA_shared_layers_ids_list)]
    args.LoRA_domain_specfic_layers_ids_list = [str(i) for i in sorted(args.LoRA_domain_specfic_layers_ids_list)]
    args.chosen_layers_for_intermediate_domain_classifiers = [str(i) for i in sorted(args.chosen_layers_for_intermediate_domain_classifiers)]
    

def extract_enums(args):
    items = args.items()
    
    def convert_string_value_to_enum(value: str):
        result = value
        
        if isinstance(value, str):
            if '.' in value and len(value.split('.')) == 2:    # Enums
                enum_name_str, enum_value_str = value.split('.')
                enum_class = globals().get(enum_name_str)
                if enum_class is not None:
                    result = enum_class[enum_value_str]
        
        return result
    
    for k, v in items:
        if isinstance(v, list):
            converted_list = []
            for v2 in v:
                converted = convert_string_value_to_enum(v2)
                converted_list.append(converted)
            args[k] = converted_list
        else:
            args[k] = convert_string_value_to_enum(v)
            

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorithms.')
    parser.add_argument('config', type=str, help='Json file of settings.')
    # Setting the order or device in a JSON file will override its values from the terminal.
    parser.add_argument('-order', type=int, help='Order of domains (it can be set on the JSON file as well)')
    parser.add_argument('-device', type=int, default=0, help='Device')
    return parser


if __name__ == '__main__':
    main()
