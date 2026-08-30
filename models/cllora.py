import gc
import logging

import numpy as np
import torch
from box import Box
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import our_utils as ou
from models.base import BaseLearner
from models.cllora_net import CLLoRANet
from utils.toolkit import accuracy_domain_shot, tensor2numpy


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def compute_orthogonality_loss(
    previous_weights_list,
    current_weights,
    epsilon=1e-8
):
    total_ortho_loss = 0.0

    current_flat = current_weights.flatten()
    current_normalized = current_flat / (torch.norm(current_flat) + epsilon)

    for prev_weights in previous_weights_list:
        prev_flat = prev_weights.flatten().to(current_normalized.device)
        prev_normalized = prev_flat / (torch.norm(prev_flat) + epsilon)
        total_ortho_loss = total_ortho_loss + torch.abs(torch.sum(prev_normalized * current_normalized))

    if len(previous_weights_list) > 0:
        total_ortho_loss = total_ortho_loss / len(previous_weights_list)

    return total_ortho_loss


class Learner(BaseLearner):
    def __init__(self, args: Box):
        super().__init__(args)

        self._network: CLLoRANet = CLLoRANet(args)

        self.args = args
        self.class_num = self._network.class_num

        self.batch_size = args.batch_size
        self.lr_default = args.lr_default
        self.lr_LoRAs = args.lr_LoRAs
        self.lr_classifier = args.lr_classifier
        self.weight_decay = args.weight_decay
        self.min_lr = args.min_lr
        self.init_cls = args.init_cls
        self.inc = args.increment

        self.topk = 2

        self.lambda_1 = args.lambda_1                          # KD ratio
        self.lambda_orthogonality = args.lambda_orthogonality

        self._cur_domain_id = -1
        self.total_sessions = args.total_sessions

        self.use_init_ptm = args.use_init_ptm
        self.use_distillation = args.use_distillation
        self.use_block_weight = args.use_block_weight

        self.data_manager = None
        self.train_dataset = self.train_loader = None
        self.test_dataset = self.test_loader = None
        self.train_dataset_for_protonet = self.train_loader_for_protonet = None

    # boundaries
    def after_task(self):
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()

    # data
    def _prepare_the_dataloaders(self, data_manager):
        self.data_manager = data_manager

        args_common = dict(batch_size=self.batch_size, num_workers=self.args.num_workers, shuffle=True)

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(self.train_dataset, drop_last=True, **args_common)

        self.test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(self.test_dataset, drop_last=False, **args_common)

        # Prototype extraction uses the *test* transforms on the training images.
        self.train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, drop_last=False, **args_common)

    # training
    def incremental_train(self, data_manager):
        self._cur_domain_id += 1

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_domain_id)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self._prepare_the_dataloaders(data_manager)

        self._network.initialize_new_classifiers_for_new_domain(nb_classes=self._total_classes)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self._network.add_fc()

        self.replace_fc(self.train_loader_for_protonet)

        gc.collect()

    def _train(self, train_loader):
        self._network.to(device=self._device)

        assert self.init_cls == self.inc, 'DIL: one domain == the full label set.'

        optimizer = self.get_optimizer_training()
        scheduler = self.get_scheduler(optimizer, self.args.epochs)

        self._init_train(train_loader=train_loader, optimizer=optimizer, scheduler=scheduler)

    def get_optimizer_training(self):
        """Two parameter groups (LoRAs + block weights, and the proxy head)."""
        params_list = ou.get_params_groups(
            model=self._network.backbone, name_model='LoRAs', lr=self.lr_LoRAs, weight_decay=self.weight_decay)

        params_list += ou.get_params_groups(
            model=self._network.proxy_fc, name_model='proxy_classifier', lr=self.lr_classifier,
            weight_decay=self.args.weight_decay_classifiers)

        ou.show_number_of_parameters_in_pramas_groups(params_list, logging)

        return ou.get_optimizer_from_params(
            params_all=params_list, optimizer_name=self.args.optimizer,
            lr_default=self.lr_default, weight_decay=self.weight_decay)

    def get_scheduler(self, optimizer, epoch):
        if self.args.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args.scheduler == 'steplr':
            return optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args.init_milestones, gamma=self.args.lr_default_decay)
        elif self.args.scheduler == 'constant':
            return None
        raise NotImplementedError(f'Unknown scheduler: {self.args.scheduler}')

    def _init_train(self, train_loader, optimizer, scheduler):
        epochs = self.args.epochs

        prog_bar = tqdm(range(epochs), desc='Epoch')

        logging.info(f'Training the domain {self._cur_domain_id + 1} / {self.total_sessions} ...')

        backbone = self._network.backbone

        for epoch in prog_bar:
            self._network.train()

            losses = 0.0
            correct, total = 0, 0

            for batch in train_loader:
                batch = ou.to_device(batch, self._device)
                _, inputs, targets = batch

                targets_from_zero = targets % self.class_num

                if self._cur_domain_id > 0 and self.use_distillation:
                    out_new, out_teacher = self._network.forward_kd(inputs, self._cur_domain_id)
                    loss_kd = self.lambda_1 * _KD_loss(out_new["logits"], out_teacher["logits"],
                                                       T=self.args.kd_temperature)

                    optimizer.zero_grad()
                    loss_kd.backward()

                    # Re-weight the gradient of the shared up-projection by the
                    # row-norms of the previous domain's up-projection.
                    for block_id in backbone.general_pos:
                        pos = backbone.adapt_pos.index(block_id)
                        for jj in range(len(backbone.msa)):
                            if backbone.msa[jj] != 1:
                                continue

                            old_A = backbone.old_adapter_list[self._cur_domain_id - 1][pos][jj].lora_A.weight
                            cur_A = backbone.cur_adapter[pos][jj].lora_A.weight

                            if cur_A.grad is None:
                                continue

                            temp_weights = torch.norm(old_A, dim=1)
                            temp_weights = len(temp_weights) * temp_weights / torch.sum(temp_weights)

                            cur_A.grad = temp_weights.unsqueeze(1) * cur_A.grad

                    optimizer.step()

                # Classification on the current-domain proxy head
                output = self._network.forward(inputs, test=False)
                logits = output["logits"]

                loss = F.cross_entropy(logits, targets_from_zero)

                # Orthogonality between the block weights
                if self._cur_domain_id > 0 and self.use_block_weight:
                    orth_loss_specific = compute_orthogonality_loss(
                        backbone.block_weight_list, backbone.block_weight)
                    loss = loss + self.lambda_orthogonality * orth_loss_specific

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets_from_zero.expand_as(preds)).cpu().sum()
                total += len(targets_from_zero)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
            #     self._cur_domain_id, epoch + 1, epochs, losses / len(train_loader), train_acc)
            info = f"Domain {self._cur_domain_id + 1}/{self.total_sessions}, Epoch {epoch + 1}/{epochs} => Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.2f}"
            prog_bar.set_description(info)
            logging.info(info)

    # ------------------------------------------------------------ prototypes
    @torch.no_grad()
    def replace_fc(self, loader):
        model = self._network
        model.eval()

        logging.info('Computing the class prototypes ...')

        start_index = -1 if self.use_init_ptm else 0

        for index in range(start_index, self._cur_domain_id + 1):
            embeddings_list, labels_list = [], []

            for batch in loader:
                batch = ou.to_device(batch, self._device)
                _, inputs, labels = batch

                embeddings = model.backbone.forward_proto(inputs, adapt_index=index)

                embeddings_list.append(embeddings.cpu())
                labels_list.append(labels.cpu())

            embeddings_all = torch.cat(embeddings_list, dim=0)
            labels_all = torch.cat(labels_list, dim=0)

            # index == -1 (the raw ViT) is stored in the very first block.
            block_index = index + 1 if self.use_init_ptm else index

            for class_index in np.unique(labels_all.numpy()):
                data_index = (labels_all == class_index).nonzero().squeeze(-1)
                proto = embeddings_all[data_index].mean(0)

                model.fc.weight.data[
                    class_index, block_index * model.out_dim:(block_index + 1) * model.out_dim
                ] = proto.to(model.fc.weight.device)

            del embeddings_list, labels_list, embeddings_all, labels_all
            gc.collect()

    # ------------------------------------------------------------ evaluation
    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy_domain_shot(
            y_pred.T[0],
            y_true,
            self._known_classes,
            class_num=self.class_num,
            many_shot=self.data_manager.many_shot_classes,
            medium_shot=self.data_manager.medium_shot_classes,
            few_shot=self.data_manager.few_shot_classes,
        )
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        return ret

    def eval_task(self):
        (predicted_labels_without_oracle, predicted_labels_with_oracle,
         true_labels, domain_classification_accuracy) = self._eval_cnn(self.test_loader)

        accuracies_without_oracle_dict = self._evaluate(predicted_labels_without_oracle, true_labels)

        accuracies_with_oracle_dict = {}
        if self.args.dataset != 'core50':
            accuracies_with_oracle_dict = self._evaluate(predicted_labels_with_oracle, true_labels)

        return accuracies_without_oracle_dict, accuracies_with_oracle_dict, domain_classification_accuracy

    @torch.no_grad()
    def _eval_cnn(self, loader):
        logging.info("Evaluating on the test set ...")
        self._network.eval()

        predicted_labels_without_oracle_list = []
        predicted_labels_with_oracle_list = []
        true_labels_list = []

        is_core50 = self.args.dataset == 'core50'

        domain_classification_accuracy_calculator = ou.AverageAccuracyCalculator()
        domain_classification_accuracy = 0.0

        for batch in loader:
            batch = ou.to_device(batch, self._device)
            _, inputs, labels = batch

            logits = self._network.forward(inputs, test=True)['logits']

            predicted_labels_without_oracle = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
            predicted_labels_without_oracle_list.append(predicted_labels_without_oracle.cpu().numpy())

            if not is_core50:
                domain_ids_ground_truth = labels // self.class_num

                # The implicitly predicted domain id.
                domain_ids_predicted = logits.argmax(dim=1) // self.class_num
                domain_classification_accuracy_calculator.update(domain_ids_predicted, domain_ids_ground_truth)

                # "With oracle": restrict the logits to the ground-truth domain block.
                logits_with_oracle = self._mask_logits_to_domains(logits, domain_ids_ground_truth)

                predicted_labels_with_oracle = torch.topk(
                    logits_with_oracle, k=self.topk, dim=1, largest=True, sorted=True)[1]
                predicted_labels_with_oracle_list.append(predicted_labels_with_oracle.cpu().numpy())

            true_labels_list.append(labels.cpu().numpy())

        predicted_labels_with_oracle = None
        if not is_core50:
            domain_classification_accuracy = domain_classification_accuracy_calculator.calculate()
            predicted_labels_with_oracle = np.concatenate(predicted_labels_with_oracle_list)

        return (
            np.concatenate(predicted_labels_without_oracle_list),
            predicted_labels_with_oracle,
            np.concatenate(true_labels_list),
            domain_classification_accuracy,
        )

    def _mask_logits_to_domains(self, logits: torch.Tensor, domain_ids: torch.Tensor):
        """Keep only the class_num logits that belong to the given domain."""
        mask = torch.full_like(logits, float('-inf'))

        num_domains = logits.shape[1] // self.class_num
        domain_ids = domain_ids.clamp(max=num_domains - 1)

        offsets = domain_ids.unsqueeze(1) * self.class_num
        columns = offsets + torch.arange(self.class_num, device=logits.device).unsqueeze(0)

        mask.scatter_(1, columns, logits.gather(1, columns))

        return mask
