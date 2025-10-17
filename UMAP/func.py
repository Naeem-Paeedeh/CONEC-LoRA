import torch
from torch.utils.data import DataLoader
from torch import Tensor as T
import numpy as np

from box import Box
from main import load_json, extract_enums, verify_setting, check_required_arguments, set_default_values
from trainer import _train
from models.conec_lora import Learner
from collections import defaultdict


def load_model(config_path: str, saved_file_path: str):
    args = Box()
    param = load_json(config_path)
    args = vars(args)  # Converting argparse Namespace to a dict.

    args = Box(args)

    args.update(param)  # Add parameters from json
        
    extract_enums(args)

    verify_setting(args)

    args.config = config_path

    args.order = 1

    check_required_arguments(args)

    set_default_values(args)

    args.epochs = 0
    args.epochs_domain_classifier_training = 0
    args.save_last_model = False
    args.save_model_after_each_task = False
    args.seed = 0
    args.device = 0
    args.UMAP = True
    
    # Initializing everthing again:
    m = _train(args)
    
    device = args.device[0]
    m_saved = Box(torch.load(saved_file_path, map_location=m._device))
    m._network.load_state_dict(m_saved.model_state)
    m._network.backbone = m._network.backbone.to(device)
    m._network.backbone = m._network.backbone.eval()
    
    from utils.data_manager import DataManager

    data_manager = DataManager(
        args.dataset,
        args.shuffle,
        args.seed,
        args.init_cls,
        args.increment,
        args,
    )
    
    args_common_data_loader = dict(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    test_dataset = data_manager.get_dataset(np.arange(0, m._total_classes), source="test", mode="test")

    loader_test = DataLoader(test_dataset, drop_last=False, **args_common_data_loader)
    
    return m, loader_test
    
 
@torch.no_grad()
def obtain_embeddings(m: Learner, loader_test, device, num_samples_per_class: int = -1):
    def init_empty_tensor_float32() -> T:
        return torch.Tensor().to(dtype=torch.float32)
    
    embeddings_before_training_all = torch.Tensor().to(dtype=torch.float32)
    embeddings_after_training_all = torch.Tensor().to(dtype=torch.float32)
    labels_all = torch.Tensor().to(dtype=torch.long)
    
    count = defaultdict(int)
    
    total_classes = m.total_sessions * m.class_num
    
    m._network.eval()
    
    for iter_num, batch in enumerate(loader_test):
        _, x, labels = batch
        
        x = x.to(device)
        
        assert x.dim() == 4
        
        final_batch = False
        
        if num_samples_per_class > 0:
            c = 0
            
            x_chosen = torch.Tensor().to(dtype=torch.float32, device=device)
            labels_chosen = torch.Tensor().to(dtype=torch.long)
            
            for lbl in range(total_classes):
                mask = labels == lbl
                
                if count[lbl] < num_samples_per_class and sum(mask) > 0:
                    samples_current_class = x[mask]
                    if samples_current_class.dim() == 3:
                        samples_current_class_chosen = samples_current_class.unsqueeze(0)
                        
                    samples_current_class_chosen = samples_current_class[:num_samples_per_class - count[lbl]]
                    
                    if samples_current_class_chosen.dim() == 3:
                        samples_current_class_chosen = samples_current_class_chosen.unsqueeze(0)
                        
                    assert x.dim() == 4
                
                    x_chosen = torch.cat([x_chosen, samples_current_class_chosen])
                    
                    labels_current_class = labels[mask]
                    labels_current_class_chosen = labels_current_class[:num_samples_per_class - count[lbl]]
                    labels_chosen = torch.cat([labels_chosen, labels_current_class_chosen])
                    
                    count[lbl] += len(labels_current_class)
                
                    if count[lbl] >= num_samples_per_class:
                        c += 1
                
            if c == total_classes:
                final_batch
        else:   # If num_samples_per_class is not set, we obtain the whole embeddings
            x_chosen = x
            labels_chosen = labels
            
        if x_chosen.dim() == 4:
            domain_ids_predicted = m.detect_domain_id(x_chosen)
            
            embeddings_before_training = m._network.forward_without_LoRAs(x_chosen, block_ids_to_return=['final'])['final'][:, 0]
            
            temp_dict = m._network.forward(x_chosen, test=True, domain_ids=domain_ids_predicted)
            
            embeddings_after_training = temp_dict['features']
            
            embeddings_before_training_all = torch.cat([embeddings_before_training_all, embeddings_before_training.cpu()])
            embeddings_after_training_all = torch.cat([embeddings_after_training_all, embeddings_after_training.cpu()])
            
            labels_all = torch.cat([labels_all, labels_chosen])
        
        if final_batch:
            break
    
    return embeddings_before_training_all, embeddings_after_training_all, labels_all
