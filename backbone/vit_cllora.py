import copy
import math
from collections import OrderedDict
from functools import partial

import timm
import torch
import torch.nn as nn
from box import Box
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed
from torch import Tensor as T


class Adapter_lora(nn.Module):
    def __init__(self, dim_embed: int, downsize_dimension: int, random_orth: bool = True):
        super().__init__()

        self.random_orth = random_orth
        self.n_embd = dim_embed
        self.down_size = downsize_dimension

        self.lora_A = nn.Linear(self.down_size, self.n_embd, bias=False)
        self.lora_B = nn.Linear(self.n_embd, self.down_size, bias=False)

        if self.random_orth:
            random_matrix = torch.rand(self.n_embd, self.down_size)
            q, _ = torch.linalg.qr(random_matrix)
            with torch.no_grad():
                self.lora_B.weight.copy_(q.T)
            self.lora_B.weight.data *= 1.0
        else:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))

        with torch.no_grad():
            nn.init.zeros_(self.lora_A.weight)

    def forward(self, x: T):
        return self.lora_A(self.lora_B(x))


class Attention_lora(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, msa=(0, 0, 0)):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.msa = list(msa)

    def _shape(self, tensor: T, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x: T, adapt: nn.ModuleList = None, block_weight: T = None):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if adapt is not None:
            if block_weight is None:
                block_weight = torch.ones(3, device=x.device, dtype=x.dtype)

            if self.msa[0] == 1:
                q = q + block_weight[0] * adapt[0](x)
            if self.msa[1] == 1:
                k = k + block_weight[1] * adapt[1](x)
            if self.msa[2] == 1:
                v = v + block_weight[2] * adapt[2](x)

        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention_lora(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                   proj_drop=drop, msa=config.msa)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x: T, adapt: nn.ModuleList = None, block_weight: T = None):
        x = x + self.drop_path(self.attn(self.norm1(x), adapt=adapt, block_weight=block_weight))
        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))
        x = residual + x
        return x


class VisionTransformer(nn.Module):
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 representation_size=None, distilled=False, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 tuning_config: Box = None, args: Box = None):
        super().__init__()

        self.args = args
        self.config = tuning_config
        self._device = tuning_config._device

        print("We are using ViT with CL-LoRA adapters.")

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # ---- CL-LoRA configuration (CONEC-LoRA argument names) --------------
        self.msa = list(tuning_config.msa)                      # LoRA_qkv_mask
        self.general_pos = sorted(tuning_config.general_pos)    # task-shared blocks
        self.specfic_pos = sorted(tuning_config.specfic_pos)    # task-specific blocks
        self.adapt_pos = sorted(self.general_pos + self.specfic_pos)
        self.use_distillation = tuning_config.use_distillation
        self.use_block_weight = tuning_config.use_block_weight
        self.freeze_B_matrices_in_shared_LoRAs = tuning_config.freeze_B_matrices_in_shared_LoRAs
        self.msa_adapt = True

        assert depth == len(self.adapt_pos), \
            'Every ViT block must be assigned either to the shared or to the domain-specific group.'
        assert len(set(self.adapt_pos)) == depth

        if self.use_distillation:
            self.old_adapter_list = nn.ModuleList()

        if self.use_block_weight:
            self.block_weight_list = []
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, 0.5, 1.5)

        # ---- standard ViT ---------------------------------------------------
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                  act_layer=act_layer, config=tuning_config, layer_id=i)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh()),
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        # ---- adapter bookkeeping -------------------------------------------
        self.adapter_list = []       # archived task-specific adapters (one entry per finished domain)
        self.adapter_pos_list = []
        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter_initial_msa()

    # ------------------------------------------------------------------ utils
    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def _build_attention_adapter(self):
        temp_adapter = nn.ModuleList()
        for j in self.msa:
            if j == 1:
                adapter = Adapter_lora(dim_embed=self.embed_dim,
                                       downsize_dimension=self.config.ffn_num).to(self._device)
            else:
                adapter = nn.Identity()
            temp_adapter.append(adapter)
        return temp_adapter

    def _freeze_B_matrices_in_shared_LoRAs(self):
        if not self.freeze_B_matrices_in_shared_LoRAs:
            return
        for block_id in self.general_pos:
            pos = self.adapt_pos.index(block_id)
            for j in range(len(self.msa)):
                if self.msa[j] == 1:
                    self.cur_adapter[pos][j].lora_A.requires_grad_(True)
                    self.cur_adapter[pos][j].lora_B.requires_grad_(False)

    def get_new_adapter_initial_msa(self):
        """First domain: create a LoRA for every adapted block."""
        for _ in range(len(self.adapt_pos)):
            self.cur_adapter.append(self._build_attention_adapter())
        self.cur_adapter.requires_grad_(True)
        self._freeze_B_matrices_in_shared_LoRAs()

    def get_new_adapter_msa(self):
        """Later domains: only the *task-specific* LoRAs are re-initialised."""
        for block_id in self.specfic_pos:
            pos = self.adapt_pos.index(block_id)
            self.cur_adapter[pos] = self._build_attention_adapter().requires_grad_(True)

        self.cur_adapter.requires_grad_(True)
        self._freeze_B_matrices_in_shared_LoRAs()

    def add_adapter_to_list(self):
        """Domain boundary: archive the task-specific LoRAs / block weights."""
        temp_adapter = []
        for block_id in self.specfic_pos:
            pos = self.adapt_pos.index(block_id)
            temp_adapter.append(copy.deepcopy(self.cur_adapter[pos].requires_grad_(False)))
        self.adapter_list.append(temp_adapter)

        if self.use_block_weight:
            block_weight_old = copy.deepcopy(self.block_weight)
            self.block_weight_list.append(block_weight_old.requires_grad_(False))
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos), device=self._device))
            nn.init.uniform_(self.block_weight, 0.5, 1.5)

        self.adapter_pos_list.append(self.adapt_pos)

        if self.use_distillation:
            self.old_adapter_list.append(copy.deepcopy(self.cur_adapter).requires_grad_(False))

        self.get_new_adapter_msa()

    # forward
    def _embed(self, x: T):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def _head(self, x: T):
        if self.global_pool:
            return self.fc_norm(x[:, 1:, :].mean(dim=1))
        return self.norm(x)

    def _current_block_weight(self, block_id: int, block_weight_source: T = None):
        if not (self.use_block_weight and block_id in self.specfic_pos):
            return None
        pos_spec = self.specfic_pos.index(block_id)
        source = self.block_weight if block_weight_source is None else block_weight_source
        return source[:, pos_spec]

    def forward_train(self, x: T):
        x = self._embed(x)

        for block_id, blk in enumerate(self.blocks):
            if block_id in self.adapt_pos:
                pos = self.adapt_pos.index(block_id)
                x = blk(x, self.cur_adapter[pos], block_weight=self._current_block_weight(block_id))
            else:
                x = blk(x, adapt=None, block_weight=None)

        x = self._head(x)
        return x if self.global_pool else x[:, 0]

    def _forward_with_adapter_index(self, x_init: T, adapter_index: int):
        x = x_init.clone()

        if adapter_index == -1:
            x = self.blocks(x)
            return self.norm(x)[:, 0, :]

        use_archived = adapter_index < len(self.adapter_list)

        for block_id in range(len(self.blocks)):
            if block_id not in self.adapt_pos:
                x = self.blocks[block_id](x, adapt=None, block_weight=None)
                continue

            if block_id in self.general_pos:
                # Task-shared LoRA is always the (single, continuously trained) current one.
                adapt = self.cur_adapter[self.adapt_pos.index(block_id)]
                block_weight = None
            else:
                if use_archived:
                    adapt = self.adapter_list[adapter_index][self.specfic_pos.index(block_id)]
                    source = self.block_weight_list[adapter_index] if self.use_block_weight else None
                else:
                    adapt = self.cur_adapter[self.adapt_pos.index(block_id)]
                    source = None
                block_weight = self._current_block_weight(block_id, block_weight_source=source)

            x = self.blocks[block_id](x, adapt, block_weight=block_weight)

        x = self.norm(x)
        return x[:, 0, :]

    def forward_test(self, x: T, use_init_ptm: bool = False):
        x_init = self._embed(x)

        features = []
        if use_init_ptm:
            features.append(self._forward_with_adapter_index(x_init, -1))

        for adapter_index in range(len(self.adapter_list) + 1):
            features.append(self._forward_with_adapter_index(x_init, adapter_index))

        return features

    def forward(self, x: T, test: bool = False, use_init_ptm: bool = False):
        if not test:
            return self.forward_train(x)

        features = self.forward_test(x, use_init_ptm=use_init_ptm)
        return torch.cat(features, dim=1)

    def forward_proto(self, x: T, adapt_index: int):
        x_init = self._embed(x)
        return self._forward_with_adapter_index(x_init, adapt_index)

    def forward_general_cls(self, x: T, t_idx: int):
        """Student / teacher CLS tokens obtained with only the task-shared blocks."""
        x = self._embed(x)
        x_teacher = x.clone()

        for block_id in self.general_pos:
            pos = self.adapt_pos.index(block_id)
            x = self.blocks[block_id](x, self.cur_adapter[pos])
        output_new = self.norm(x)[:, 0, :]

        for block_id in self.general_pos:
            pos = self.adapt_pos.index(block_id)
            x_teacher = self.blocks[block_id](x_teacher, self.old_adapter_list[t_idx - 1][pos])
        output_teacher = self.norm(x_teacher)[:, 0, :]

        return output_new, output_teacher


def _load_pretrained_weights(model: VisionTransformer, timm_name: str):
    checkpoint_model = timm.create_model(timm_name, pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()

    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = qkv_weight[:768]
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = qkv_weight[768:768 * 2]
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = qkv_weight[768 * 2:]
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = qkv_bias[:768]
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = qkv_bias[768:768 * 2]
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = qkv_bias[768 * 2:]

    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            state_dict[key.replace('mlp.', '')] = state_dict.pop(key)

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    for name, p in model.named_parameters():
        p.requires_grad = name in msg.missing_keys

    model._freeze_B_matrices_in_shared_LoRAs()
    return model


def vit_base_patch16_224_cllora(args: Box = None, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), args=args, **kwargs)
    return _load_pretrained_weights(model, "vit_base_patch16_224")


def vit_base_patch16_224_in21k_cllora(args: Box = None, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), args=args, **kwargs)
    return _load_pretrained_weights(model, "vit_base_patch16_224_in21k")
