import copy
import logging
import torch
from torch import nn
from backbone.linears import CosineLinearFeature
import timm
import math
from new_types import ClassifierType
from backbone import vit_conec_lora
from box import Box
import our_utils as ou
from backbone.vit_conec_lora import VisionTransformer
from torch import Tensor as T
from models.mlp import MLP
from models.domain_classifier import DomainClassifier
from models.stochastic_classifier import StochasticClassifier, Linear2


def get_backbone(args):
    name = args.backbone_type.lower()
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif '_conec_lora' in name:
        LoRA_downsize_dimension = args.LoRA_downsize_dimension
        
        if args.model_name == "conec_lora" or args.model_name == "cllora":
            tuning_config = Box(
                LoRA_qkv_mask=args.LoRA_qkv_mask,
                LoRA_domain_specfic_layers_ids_list=args.LoRA_domain_specfic_layers_ids_list,
                LoRA_shared_layers_ids_list=args.LoRA_shared_layers_ids_list,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                LoRA_downsize_dimension=LoRA_downsize_dimension,        # The down-projection dimension.
                d_model=768,
                _device=args.device[0],
            )
            
            if name == "vit_base_patch16_224_conec_lora":
                model = vit_conec_lora.vit_base_patch16_224_conec_lora(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, args=args, tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "vit_base_patch16_224_in21k_conec_lora":
                model = vit_conec_lora.vit_base_patch16_224_in21k_conec_lora(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, args=args, tuning_config=tuning_config)
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone: VisionTransformer = get_backbone(args=args)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args.device[0]

        if 'resnet' in args.backbone_type:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone.forward(x)['features']
        else:
            output = self.backbone.forward(x)
            return output

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone.forward(x)
            out = self.fc.forward(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone.forward(x)
            out = self.fc.forward(x)
            out.update({"features": x})

        return out

    def initialize_new_classifiers_for_new_domain(self, nb_classes):
        pass

    def generate_final_classifier(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze_or_unfreeze(self):
        raise NotImplementedError


class OurNet(BaseNet):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.inc = args.increment
        self.init_cls = args.init_cls
        self._cur_domain_id = -1
        self.out_dim = self.backbone.out_dim
        self.fc = None
        self.proxy_fc = None
        self.use_proxy_classifier = args.use_proxy_classifier
        
        self.class_num = 1
        if args["dataset"] == "cddb":
            self.class_num = 2
        elif args["dataset"] == "domainnet":
            self.class_num = 345
        elif args["dataset"] == "core50":
            self.class_num = 50
        elif args["dataset"] == "officehome":
            self.class_num = 65
        else:
            raise ValueError("Unknown datasets: {}.".format(args["dataset"]))
        
        self.fc: CosineLinearFeature = None
        
        self.classifiers_list = nn.ModuleList()
        self.total_sessions = args.total_sessions
        self.domain_classifier = None
        self.transformation_modules: nn.ModuleDict = None
        self.chosen_layers_for_intermediate_domain_classifiers = args.chosen_layers_for_intermediate_domain_classifiers
        self.temperature_stochastic_classifier = args.temperature_stochastic_classifier
        self.classifier_type = self.args.classifier_type
        
        self.initialize_transformation_module()
        
        self.domain_classifier = DomainClassifier(args).to(self._device)
        
    def initialize_transformation_module(self):
        if self.args.use_transformation_module:
            self.transformation_modules = nn.ModuleDict()
            
            for block_id_str in self.chosen_layers_for_intermediate_domain_classifiers:
                self.transformation_modules[block_id_str] = MLP(dim_input=self.out_dim, dim_hidden=1024, dim_output=self.out_dim, num_layers=2, dropout_rate=0.0, device=self._device)       # 120, 1024, 16, 768
        else:
            self.transformation_modules = None
    
    def freeze_or_unfreeze(self, requires_grad: bool = False):
        ou.freeze_or_unfreeze(self.domain_classifier, requires_grad=requires_grad)
        
        ou.freeze_or_unfreeze(self.transformation_modules, requires_grad=requires_grad)
        
    @property
    def feature_dim(self):
        return self.out_dim * (self._cur_domain_id + 1)

    @property
    def number_of_learned_domains(self):
        if self.domain_classifier is None:
            return 1
        else:
            return self.domain_classifier.weight.shape[0]
    
    def initialize_new_classifiers_for_new_domain(self, nb_classes):        # It was  update_fc
        self._cur_domain_id += 1
        
        assert self.init_cls == self.inc
        
        if self.use_proxy_classifier:
            self.proxy_fc = self.generate_temporary_classifier(self.out_dim, self.inc).to(self._device)
            
        if self._cur_domain_id == 0:
            self.classifiers_list.append(self.generate_final_classifier().to(self._device))
        else:
            if self.classifier_type in [ClassifierType.Separate_CosineLinearLayers, ClassifierType.Separate_LinearLayers, ClassifierType.Separate_Stochastic_Classifiers]:
                if self._cur_domain_id > 0:
                    self.classifiers_list.append(copy.deepcopy(self.classifiers_list[-1]))
                    
                    ou.freeze_or_unfreeze(self.classifiers_list[-2], requires_grad=False)
                    
            elif self.classifier_type in [ClassifierType.Single_CosineLinearFeature, ClassifierType.Single_Linear]:
                pass
            else:
                raise NotImplementedError

    def generate_temporary_classifier(self, in_dim, out_dim):
        if self.args.temporary_classifier_type == ClassifierType.Single_CosineLinearFeature:
            layer = CosineLinearFeature(in_dim, out_dim)
        elif self.args.temporary_classifier_type == ClassifierType.Single_Stochastic_Classifier:
            layer = StochasticClassifier(in_dim, out_dim, temperature=self.temperature_stochastic_classifier)
        elif self.args.temporary_classifier_type == ClassifierType.Single_Linear:
            layer = Linear2(in_dim, out_dim, bias=False)
        else:
            raise NotImplementedError
        return layer
    
    def generate_final_classifier(self):
        in_dim = self.out_dim
        
        out_dim = self.inc
            
        if self.classifier_type in [ClassifierType.Single_CosineLinearFeature, ClassifierType.Separate_CosineLinearLayers]:
            layer = CosineLinearFeature(in_dim, out_dim)
        elif self.classifier_type in [ClassifierType.Separate_Stochastic_Classifiers]:
            layer = StochasticClassifier(in_dim, out_dim)
        elif self.classifier_type in [ClassifierType.Single_Linear, ClassifierType.Separate_LinearLayers]:
            layer = Linear2(in_dim, out_dim, bias=False)
        else:
            raise NotImplementedError
        return layer
    
    def extract_vector(self, x):
        output = self.backbone.forward(x)
        return output

    def forward_kd(self, x, t_idx):
        # It forwards the same inputs through the shared LoRAs for both current domain and previous domain.
        x_new, x_previous_domain = self.backbone.forward_general_cls(x=x, domain_id=t_idx)
        
        if self.use_proxy_classifier:
            out_new_domain = self.proxy_fc.forward(x_new)
            out_previous_domain = self.proxy_fc.forward(x_previous_domain)
        else:
            out_new_domain = self.classifiers_list[-1].forward(x_new)
            out_previous_domain = self.classifiers_list[-1].forward(x_previous_domain)
        
        return out_new_domain, out_previous_domain

    def forward(self, x, test: bool = False, domain_ids: T = None):
        if not test:    # if training
            x = self.backbone.forward(x=x, test=False)
            if self.use_proxy_classifier:
                out = self.proxy_fc.forward(x)
            else:
                out = self.classifiers_list[-1].forward(x)
            out["features"] = x
            return out
        else:       # if test
            assert domain_ids is not None
            
            features = self.backbone.forward_with_chosen_domains(x=x, domain_ids=domain_ids)
            
            out = {}
            
            if self.classifier_type in [ClassifierType.Single_Linear, ClassifierType.Single_CosineLinearFeature]:
                out = self.classifiers_list[-1].forward(features)
            elif self.classifier_type in [ClassifierType.Separate_CosineLinearLayers, ClassifierType.Separate_Stochastic_Classifiers]:
                
                logits = torch.zeros([features.shape[0], self.classifiers_list[-1].out_dim], device=x.device)
                
                for domain_id_temp in domain_ids.unique().tolist():
                    mask = domain_ids == domain_id_temp
                    features_selected_domain = features[mask]
                    logits[mask] = self.classifiers_list[domain_id_temp].forward(features_selected_domain, return_dict=False)
                
                out['logits'] = logits
            elif self.classifier_type == ClassifierType.Separate_LinearLayers:
                logits = torch.zeros([features.shape[0], self.classifiers_list[-1].out_dim], device=x.device)
                
                for domain_id_temp in domain_ids.unique().tolist():
                    mask = domain_ids == domain_id_temp
                    
                    features_selected_domain = features[mask]
                    
                    dist = torch.cdist(features_selected_domain, self.classifiers_list[domain_id_temp].layer.weight, p=2)
                    
                    logits[mask] = -1.0 * dist
                    
                out['logits'] = logits
            else:
                raise NotImplementedError
            
            out['features'] = features
            return out
            
        raise NotImplementedError

    # For domain classifiers
    def forward_without_LoRAs(self, x: T, block_ids_to_return: list = []):
        return self.backbone.forward_without_LoRAs(x, block_ids_to_return=block_ids_to_return)
