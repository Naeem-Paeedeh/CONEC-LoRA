import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed
import copy
from torch import Tensor as T
from box import Box
from models.lora import LoRA_Adapter
import our_utils as ou


class Attention_lora(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., LoRA_qkv_mask=[0, 0, 0]):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.ffn_option = 'parallel'
        self.LoRA_qkv_mask = LoRA_qkv_mask

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x: T, LoRA: nn.ModuleList = None):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if LoRA is not None:
            if self.LoRA_qkv_mask[0] == 1:
                adapt_x = LoRA[0](x)
                q = q + adapt_x
            if self.LoRA_qkv_mask[1] == 1:
                adapt_x = LoRA[1](x)
                k = k + adapt_x
            if self.LoRA_qkv_mask[2] == 1:
                adapt_x = LoRA[2](x)
                v = v + adapt_x

        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention_lora(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, LoRA_qkv_mask=config.LoRA_qkv_mask)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, LoRA: nn.ModuleList = None):
        x = x + self.drop_path(
            self.attn.forward(self.norm1(x), LoRA=LoRA))
        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))
        x = residual + x
        
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, tuning_config=None, args: Box = None):
        super().__init__()
        
        self.args = args
        self._device = tuning_config._device
        self.config = tuning_config
        print("We are using ViT with LoRAs.")
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.freeze_B_matrices_in_shared_LoRAs: bool = args.freeze_B_matrices_in_shared_LoRAs

        self.LoRA_qkv_mask = self.config.LoRA_qkv_mask
        self.LoRA_shared_layers_ids_list = args.LoRA_shared_layers_ids_list
        self.LoRA_domain_specfic_layers_ids_list = args.LoRA_domain_specfic_layers_ids_list
        self.LoRA_all_layers_ids_list = self.LoRA_shared_layers_ids_list + self.LoRA_domain_specfic_layers_ids_list
        
        self.use_transformation_module = args.use_transformation_module
        
        assert depth == len(self.LoRA_all_layers_ids_list)

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # ######## MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        # f"{Layer-ID}" -> Shared LoRA
        # f"Domain-ID,{Layer-ID}" -> Domain-specific LoRA
        self.LoRAs_dict = nn.ModuleDict()
        
        self.num_blocks = len(self.blocks)
        self._cur_domain_id = 0
        
        assert len(set(args.LoRA_shared_layers_ids_list + args.LoRA_domain_specfic_layers_ids_list)) == self.num_blocks
        
        self.initialize_LoRAs(domain_id=0)

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def _freeze_or_unfreeze_LoRAs_for_a_domain(self, domain_id: int, requires_grad: bool):
        for block_id in self.LoRA_all_layers_ids_list:
            ou.freeze_or_unfreeze(self.LoRAs_dict[f'{domain_id},{block_id}'], requires_grad=requires_grad)
    
    def _initalize_an_attention_LoRA(self):
        attention_LoRA = nn.ModuleList()
        
        for j in self.LoRA_qkv_mask:
            if j == 1:
                adapter = LoRA_Adapter(
                    args=self.args,
                    dim_embed=self.embed_dim,
                    downsize_dimension=self.args.LoRA_downsize_dimension,
                ).to(self._device)
            else:
                adapter = nn.Identity()
            attention_LoRA.append(adapter)
            
        return attention_LoRA
    
    def initialize_LoRAs(self, domain_id: int):
        
        # We always keep the shared LoRAs for all domains, but we require the shared LoRAs from the previous domain for the  loss. You can simply release the memory after the calculation of the KD loss.
        
        # f"Domain-ID,{Layer-ID}" -> Domain-specific LoRA
        for block_id in self.LoRA_all_layers_ids_list:
            if domain_id == 0:
                self.LoRAs_dict[f'{domain_id},{block_id}'] = self._initalize_an_attention_LoRA()
            else:
                self.LoRAs_dict[f'{domain_id},{block_id}'] = copy.deepcopy(self.LoRAs_dict[f'{domain_id - 1},{block_id}']).requires_grad_(True)
        
        if self.freeze_B_matrices_in_shared_LoRAs:
            self._freeze_B_matrices_in_shared_LoRAs(domain_id)
            
    def _freeze_B_matrices_in_shared_LoRAs(self, domain_id: int):
        # It freezes the B matrices for shared LoRAs.
        for block_id in self.LoRA_shared_layers_ids_list:
            for j in range(len(self.LoRA_qkv_mask)):
                if self.LoRA_qkv_mask[j] == 1:
                    self.LoRAs_dict[f'{domain_id},{block_id}'][j].A.requires_grad_(True)
                    self.LoRAs_dict[f'{domain_id},{block_id}'][j].B.requires_grad_(False)
                    
    def prepre_for_new_domain(self):      # def add_adapter_to_list(
        # We freeze the LoRAs for the old domain (current domain).
        for block_id in self.LoRA_all_layers_ids_list:
            self.LoRAs_dict[f'{self._cur_domain_id},{block_id}'].requires_grad_(False)
            
        # Next, we initalize the LoRAs for the new domain
        self.initialize_LoRAs(domain_id=self._cur_domain_id + 1)

        self._cur_domain_id += 1

    def _forward_layer_with_domain_ids_list(self, x: T, block_id: int, domain_ids):
        
        if isinstance(domain_ids, int):
            res = self._forward_layer_with_a_single_domain_id(x, block_id=block_id, domain_id=domain_ids)
        elif type(domain_ids) in [T, list]:
            
            if isinstance(domain_ids, T):
                domain_ids_unique = domain_ids.unique().tolist()
            else:
                domain_ids_unique = list(set(domain_ids))
            
            res = torch.zeros_like(x)
            
            for domain_id in domain_ids_unique:
                selected_indices = torch.where(domain_ids == domain_id)
                x_selected = x[selected_indices]
                res[selected_indices] = self._forward_layer_with_a_single_domain_id(x_selected, block_id=block_id, domain_id=domain_id)
        else:
            raise NotImplementedError
        
        return res
    
    def _forward_layer_with_a_single_domain_id(self, x: T, block_id: int, domain_id: int = -1):
        block_id_str = str(block_id)
        
        arguments_other = dict(LoRA=None)
        
        if domain_id == -1:
            domain_id = self._cur_domain_id
            
        if block_id_str in self.LoRA_all_layers_ids_list:
            arguments_other['LoRA'] = self.LoRAs_dict[f'{domain_id},{block_id_str}']

        else:
            raise NotImplementedError
                
        x = self.blocks[int(block_id)].forward(x, **arguments_other)
        
        return x
    
    # For domain classifiers
    def forward_without_LoRAs(self, x: T, block_ids_to_return: list = []):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        features_chosen_layers = {}
        
        for block_id in range(len(self.blocks)):
            x = self.blocks[int(block_id)].forward(x)
            
            block_id_str = str(block_id)
            if block_id_str in block_ids_to_return:
                features_chosen_layers[block_id_str] = x
            
        x = self.norm(x)
        features_chosen_layers['final'] = x     # [batch_size, seq_len, dim_embed]
        
        return features_chosen_layers
    
    def forward_with_chosen_domains(self, x: T, domain_ids=-1):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block_id in range(len(self.blocks)):
            if isinstance(domain_ids, int):
                x = self._forward_layer_with_a_single_domain_id(x=x, block_id=block_id, domain_id=domain_ids)
            else:
                x = self._forward_layer_with_domain_ids_list(x=x, block_id=block_id, domain_ids=domain_ids)
            
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
                    
        return outcome
    
    def forward_all_domains(self, x) -> list:
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)
        
        features = []
        
        # For all learned domains.
        for domain_id in range(self._cur_domain_id + 1):
            x = copy.deepcopy(x_init)
            for block_id in range(len(self.blocks)):

                x = self._forward_layer_with_a_single_domain_id(x, block_id=block_id, domain_id=domain_id)

            x = self.norm(x)
            cls_token = x[:, 0]
            features.append(cls_token)

        return features

    def forward(self, x, test=False):
        if not test:
            output = self.forward_with_chosen_domains(x)
            return output
        else:       # test is True
            raise NotImplementedError
            
    def forward_general_cls(self, x, domain_id: int):
        # It forward the blocks with general LoRAs and returns the CLS token.
        
        assert domain_id > 0
        
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_previous_domain = copy.deepcopy(x)        # x_teacher

        for block_id in self.LoRA_shared_layers_ids_list:
            x = self._forward_layer_with_a_single_domain_id(x=x, block_id=block_id)

        x = self.norm(x)
        cls_token_new_domain = x[:, 0, :]

        for block_id in self.LoRA_shared_layers_ids_list:
            x_previous_domain = self._forward_layer_with_a_single_domain_id(x=x_previous_domain, block_id=block_id, domain_id=domain_id - 1)
        
        x_previous_domain = self.norm(x_previous_domain)
        cls_token_previous_domain = x_previous_domain[:, 0, :]

        return cls_token_new_domain, cls_token_previous_domain


def vit_base_patch16_224_conec_lora(args=None, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), args=args, **kwargs)
    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768: 768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768: 768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    for block_id in model.LoRA_shared_layers_ids_list:
        for j in range(len(model.LoRA_qkv_mask)):
            if model.LoRA_qkv_mask[j] == 1:
                for param in model.LoRAs_dict[f'0,{block_id}'][j].B.parameters():
                    param.requires_grad = False
    return model


# def vit_base_patch16_224_in21k_conec_lora(pretrained=False, args=None, **kwargs):
    
#     model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), args=args, **kwargs)

#     checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
#     state_dict = checkpoint_model.state_dict()
#     for key in list(state_dict.keys()):
#         if 'qkv.weight' in key:
#             qkv_weight = state_dict.pop(key)
#             q_weight = qkv_weight[:768]
#             k_weight = qkv_weight[768: 768 * 2]
#             v_weight = qkv_weight[768 * 2:]
#             state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
#             state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
#             state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
#         elif 'qkv.bias' in key:
#             qkv_bias = state_dict.pop(key)
#             q_bias = qkv_bias[:768]
#             k_bias = qkv_bias[768: 768 * 2]
#             v_bias = qkv_bias[768 * 2:]
#             state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
#             state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
#             state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
#     # second, modify the mlp.fc.weight to match fc.weight
#     for key in list(state_dict.keys()):
#         if 'mlp.fc' in key:
#             fc_weight = state_dict.pop(key)
#             state_dict[key.replace('mlp.', '')] = fc_weight

#     msg = model.load_state_dict(state_dict, strict=False)
#     print(msg)

#     # freeze all but the adapter
#     for name, p in model.named_parameters():
#         if name in msg.missing_keys:
#             p.requires_grad = True
#         else:
#             p.requires_grad = False

#     if not model.msa_adapt:
#         for adapter_temp in model.cur_adapters:
#             # for adapter in adapter_temp:
#             for param in adapter_temp.B.parameters():
#                 param.requires_grad = False
#     else:
#         for i in model.adapters_all_positions:
#             # if i in model.LoRA_shared_layers_ids_list:
#             if i in model.LoRA_shared_layers_ids_list:
#                 pos = model.adapters_all_positions.index(i)
#                 for j in range(len(model.LoRA_qkv_mask)):
#                     if model.LoRA_qkv_mask[j] == 1:
#                         # for adapter in adapter_temp:
#                         for param in model.cur_adapters[pos][j].B.parameters():
#                             param.requires_grad = False

#     return model
