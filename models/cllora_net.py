import copy

import torch
from box import Box
from torch import Tensor as T
from torch import nn

from backbone.linears import CosineLinearFeature
from models.our_net import BaseNet


# Number of classes per domain for every supported DIL benchmark.
CLASS_NUM_PER_DATASET = {
    "cddb": 2,
    "domainnet": 345,
    "core50": 50,
    "officehome": 65,
}


class CLLoRANet(BaseNet):
    def __init__(self, args: Box):
        super().__init__(args)

        self.args = args
        self.inc = args.increment
        self.init_cls = args.init_cls
        self._cur_domain_id = -1
        self.out_dim = self.backbone.out_dim

        if args.dataset not in CLASS_NUM_PER_DATASET:
            raise ValueError("Unknown dataset: {}.".format(args.dataset))
        self.class_num = CLASS_NUM_PER_DATASET[args.dataset]

        assert self.init_cls == self.inc == self.class_num, \
            'For DIL, one task is one domain over the full label set.'

        self.use_init_ptm = args.use_init_ptm
        self.total_sessions = args.total_sessions

        self.fc: CosineLinearFeature = None        # block-diagonal head over all (domain, class) pairs
        self.proxy_fc: CosineLinearFeature = None  # per-domain head used during training
        self.fc_list = nn.ModuleList()             # archived per-domain proxy heads

    # ------------------------------------------------------------------ utils
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    @property
    def feature_dim(self):
        # One out_dim-sized block per learned domain (+1 for the raw ViT features).
        if self.use_init_ptm:
            return self.out_dim * (self._cur_domain_id + 2)
        return self.out_dim * (self._cur_domain_id + 1)

    def generate_fc(self, in_dim, out_dim):
        return CosineLinearFeature(in_dim, out_dim)

    # --------------------------------------------------------- domain updates
    def initialize_new_classifiers_for_new_domain(self, nb_classes: int):
        """Called once per domain, before training (CL-LoRA's update_fc)."""
        self._cur_domain_id += 1

        # Training head: only the class_num classes of the current domain.
        self.proxy_fc = self.generate_fc(self.out_dim, self.class_num).to(self._device)

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()

        if self.fc is not None:
            old_nb_classes = self.fc.out_dim
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            # Old rows keep their old feature blocks; the newly appended block starts at zero.
            fc.weight.data[:old_nb_classes, :-self.out_dim] = weight

        del self.fc
        self.fc = fc
        self.fc.requires_grad_(False)

    def add_fc(self):
        """Archive the per-domain training head at the end of a domain."""
        self.fc_list.append(self.proxy_fc.requires_grad_(False))
        self.proxy_fc = None

    # ---------------------------------------------------------------- forward
    def extract_vector(self, x: T):
        return self.backbone.forward(x)

    def forward_kd(self, x: T, domain_id: int):
        """Student / teacher logits through the *task-shared* LoRAs only."""
        x_new, x_teacher = self.backbone.forward_general_cls(x, domain_id)
        return self.proxy_fc.forward(x_new), self.proxy_fc.forward(x_teacher)

    def forward(self, x: T, test: bool = False):
        if not test:
            features = self.backbone.forward(x, test=False)
            out = self.proxy_fc.forward(features)
            out.update({"features": features})
            return out

        features = self.backbone.forward(x, test=True, use_init_ptm=self.use_init_ptm)
        out = self.fc.forward_diagonal(
            features,
            cur_task=self._cur_domain_id,
            init_cls=self.init_cls,
            inc=self.inc,
            out_dim=self.out_dim,
            use_init_ptm=self.use_init_ptm,
        )
        out.update({"features": features})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())
