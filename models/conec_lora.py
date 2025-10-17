import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.our_net import OurNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy_domain_shot
from box import Box
from new_types import (ClassifierType, GenerationStrategy)
import our_utils as ou
from torch import Tensor as T
from models.ball_generator import ball_loss_fast
from collections import defaultdict
from models.domain_classifier import DomainClassifier, DomainClassifierData
from models.gmm import GMM_EM, random_sampling
import gc


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network: OurNet = OurNet(args)
        
        self.class_num = self._network.class_num

        self.args = args
        self.batch_size = args.batch_size
        self.lr_default = args.lr_default
        self.init_cls = args.init_cls
        self.inc = args.increment

        self.topk = 2  # origin is 5
        
        self.lambda_1 = self.args.lambda_1
        self.lambda_2 = self.args.lambda_2
        
        self._cur_domain_id = -1
        self.data_manager = None
        self.train_dataset_for_protonet = None
        self.train_loader_for_protonet = None
        self.test_dataset = None
        self.test_loader = None
        
        self.classifier_type = args.classifier_type
        
        self.use_proxy_classifier = args.use_proxy_classifier
        self.LoRA_shared_layers_ids_list = args.LoRA_shared_layers_ids_list
        self.LoRA_domain_specfic_layers_ids_list = args.LoRA_domain_specfic_layers_ids_list
        self.LoRA_all_layers_ids_list = self.LoRA_shared_layers_ids_list + self.LoRA_domain_specfic_layers_ids_list
        self.chosen_layers_for_intermediate_domain_classifiers = args.chosen_layers_for_intermediate_domain_classifiers
        
        self.generation_strategy = args.generation_strategy
        
        self.total_sessions = args.total_sessions
        
        self.lr_domain_classifiers = args.lr_domain_classifiers
        self.lr_transformation_module = args.lr_transformation_module
        self.lr_LoRAs = args.lr_LoRAs
        
        self.domain_classifier_data = DomainClassifierData(device=self._device)
        self.cache_synthetic_embeddings = args.cache_synthetic_embeddings
        
        self.use_transformation_module = args.use_transformation_module
        self.confidence_threshold = args.confidence_threshold
        self.weight_decay_domain_classifiers = args.weight_decay_domain_classifiers
        self.n_components = args.n_components
        self.max_iter_for_GMM = args.max_iter_for_GMM
        self.tol_for_GMM = args.tol_for_GMM
        self.max_number_of_embeddings_in_memory = args.max_number_of_embeddings_in_memory
        
        self.temporary_classifier_type = args.temporary_classifier_type
        
        self.epochs_domain_classifier_training = args.epochs_domain_classifier_training
        
    def after_task(self):
        self._known_classes = self._total_classes
        self._network.backbone.prepre_for_new_domain()

    def _prepare_the_dataloaders(self, data_manager):
        self.data_manager = data_manager
        
        args_common_data_loader = dict(batch_size=self.batch_size, num_workers=self.args.num_workers, shuffle=True)
        
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(self.train_dataset, drop_last=True, **args_common_data_loader)
        
        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(self.test_dataset, drop_last=False, **args_common_data_loader)

        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, drop_last=False, **args_common_data_loader)
        
    def incremental_train(self, data_manager):
        self._cur_domain_id += 1
        
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_domain_id)
        
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        
        if not self.args.UMAP:
            self._prepare_the_dataloaders(data_manager)
        
        self._network.initialize_new_classifiers_for_new_domain(nb_classes=self._total_classes)
        
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        if not self.args.UMAP:
            self._train(self.train_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        
        if self.args.classifier_type in [ClassifierType.Separate_CosineLinearLayers, ClassifierType.Separate_Stochastic_Classifiers, ClassifierType.Separate_LinearLayers] and self.use_proxy_classifier:
            self.replace_fc_weight_with_prototypes(self.train_loader_for_protonet)
            
            gc.collect()
                
        if not self.args.UMAP:
            self._train_domain_classifier(self.train_loader_for_protonet)
            gc.collect()
        
        if self.use_proxy_classifier:
            del self._network.proxy_fc
            gc.collect()
        
    def _train(self, train_loader):
        self._network.to(device=self._device)
        
        if self._cur_domain_id == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer_training()
            scheduler = self.get_scheduler(optimizer, self.args.epochs)
        else:
            raise NotImplementedError

        self._init_train(train_loader=train_loader, optimizer=optimizer, scheduler=scheduler)

    def _init_train(self, train_loader, optimizer, scheduler):
        assert self.init_cls == self.inc
        
        if self._cur_domain_id == 0 or self.init_cls == self.inc:
            epochs = self.args.epochs
        else:
            raise NotImplementedError
            epochs = self.args.later_epochs
        
        prog_bar = tqdm(range(epochs), desc='Epoch')
        
        logging.info(f'Training the domain {self._cur_domain_id} ...')
        
        for epoch in prog_bar:
            self._network.train()
            
            losses = 0.0
            correct, total = 0, 0

            # for name, param in self._network.backbone.cur_adapters[0][1].named_parameters():
            #     print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
            # for name, param in self._network.backbone.cur_adapters[-1][1].named_parameters():
            #     print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")

            for batch_index, batch in enumerate(train_loader):
                batch = ou.to_device(batch, self._device)
                _, inputs, targets = batch
                
                output = self._network.forward(inputs, test=False)

                logits = output["logits"]
                
                targets_from_zero = targets % self.class_num
                
                loss = 0.0
                
                loss_ce_classification = F.cross_entropy(logits, targets_from_zero)  # From equation 12

                loss = loss + loss_ce_classification
                
                # Knowledge-distillation loss
                if self._cur_domain_id > 0:
                    
                    # We forward the inputs with the shared LoRAs for the current domain and previous domain.
                    out_new, out_teacher = self._network.forward_kd(inputs, self._cur_domain_id)
                    out_new_logits = out_new["logits"]
                    out_teacher_logits = out_teacher["logits"]
                    loss_kd = self.lambda_1 * _KD_loss(out_new_logits, out_teacher_logits, T=self.args.kd_temperature)
                    
                    optimizer.zero_grad()
                    loss_kd.backward()
                    
                    for block_id in self._network.backbone.LoRA_shared_layers_ids_list:
                        
                        for jj in range(len(self._network.backbone.LoRA_qkv_mask)):
                            if self._network.backbone.LoRA_qkv_mask[jj] == 1:
                                temp_weights = 1. * torch.norm(self._network.backbone.LoRAs_dict[f'{self._cur_domain_id - 1},{block_id}'][jj].A.weight, dim=1)
                                
                                temp_weights = 1. * len(temp_weights) * temp_weights / torch.sum(temp_weights)
                                
                                self._network.backbone.LoRAs_dict[f'{self._cur_domain_id},{block_id}'][jj].A.weight.grad = temp_weights.unsqueeze(1) * self._network.backbone.LoRAs_dict[f'{self._cur_domain_id},{block_id}'][jj].A.weight.grad
                    
                    optimizer.step()
                
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

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_domain_id,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

            logging.info(info)
    
    def _obtain_optimizer_and_scheduler_for_domain_classifier(self):
        params_list = []
        
        for block_id_str, classifier in self._network.domain_classifier.classifiers_dict.items():
            params_list += ou.get_params_groups(model=classifier, lr=self.lr_domain_classifiers, name_model='domain_classifier', weight_decay=self.weight_decay_domain_classifiers)
        
        if self.use_transformation_module:
            for block_id_str, mlp in self._network.transformation_modules.items():
                params_list += ou.get_params_groups(model=mlp, name_model='transformation_module', lr=self.lr_transformation_module, weight_decay=self.args.weight_decay_transformation_module)
        
        optimizer_domain_classifier = ou.get_optimizer_from_params(params_all=params_list, optimizer_name=self.args.optimizer, lr_default=self.args.lr_default, weight_decay=self.args.weight_decay)
        
        scheduler_domain_classifier = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_domain_classifier, T_max=self.epochs_domain_classifier_training)
        
        return optimizer_domain_classifier, scheduler_domain_classifier
    
    def get_optimizer_training(self):
        
        params_all = []
        
        lr_default = self.args.lr_default
        
        params_all += ou.get_params_groups(model=self._network.backbone, name_model='LoRAs', lr=self.lr_LoRAs)
        
        if self.use_proxy_classifier:
            params_all += ou.get_params_groups(model=self._network.proxy_fc, name_model='proxy_fc', lr=self.args.lr_temporary_classifier, weight_decay=self.args.weight_decay_classifiers)
        else:
            params_all += ou.get_params_groups(model=self._network.classifiers_list[-1], name_model='classifier', lr=self.args.lr_classifier, weight_decay=self.args.weight_decay_classifiers)
            
        ou.show_number_of_parameters_in_pramas_groups(params_all=params_all, logger=logging)
        
        optimizer = ou.get_optimizer_from_params(params_all=params_all, optimizer_name=self.args.optimizer, lr_default=lr_default, weight_decay=self.args.weight_decay)
        
        return optimizer
    
    @torch.no_grad()
    def obtain_embeddings_for_stats_calculations(self, dataloader):
        """This method obtains embeddings for all samples with frozen backbone (without the LoRAs)."""
        
        logging.info(f'Obtaining the embeddings for domain {self._cur_domain_id} ...')
        
        embeddings_dict_of_lists = defaultdict(list)
        
        cnt = 0
        
        for batch_index, (_, inputs, labels) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(self._device)
            
            embeddings_dict = self.forward_without_LoRAs(inputs)
            
            cnt += inputs.shape[0]
                
            embeddings_dict = self.process_embeddings_dict(embeddings_dict)
                
            embeddings_dict = ou.to_device(embeddings_dict, device='cpu')
            
            for block_id, embeddings in embeddings_dict.items():
                embeddings_dict_of_lists[block_id].append(embeddings)
            
            if cnt > self.max_number_of_embeddings_in_memory // len(self.chosen_layers_for_intermediate_domain_classifiers):
                break
                
        embeddings_dict = {}
        
        for block_id, embeddings_list in embeddings_dict_of_lists.items():
            embeddings_dict[block_id] = torch.cat(embeddings_list, 0).detach()
            
        return embeddings_dict
    
    def process_embeddings_dict(self, embeddings_dict: dict, normalize=False):
        """In this method, we calculate the global pool of the non-CLS tokens or the CLS token to return."""
        # We should decide whether we want to use the CLS token or the average pooling!
        embeddings_dict_result = {}
        
        for block_id, embeddings in embeddings_dict.items():
            if embeddings.dim() == 3:   # [batch_size, seq_len, dim_embed]
                # Its shape becomes: [batch_size, tokens_sequences_length - 1, dim_embed]
                embeddings = embeddings[:, 1:]        # In this case, we want to use the statistics of all tokens, excluding the CLS token.
                embeddings = embeddings.mean(1)
                # feature = eo.rearrange(feature, 'b s d -> (b s) d')
            if embeddings.dim() == 2:   # # [batch_size, dim_embed]
                pass    # We do not need anything to do
            else:
                raise NotImplementedError
            
            if normalize:
                embeddings = F.normalize(embeddings, dim=-1)
                
            embeddings_dict_result[block_id] = embeddings
        
        return embeddings_dict_result
    
    @torch.no_grad()
    def obtain_embeddings_for_a_domain_or_domains(self, loader, domain_ids=-1):
        
        if isinstance(domain_ids, int):
            logging.info(f"Obtaining the embeddings for domain {domain_ids}")
        
        model = self._network
        model = model.eval()

        embeddings_final = torch.Tensor().to(dtype=torch.float32)
        labels_final = torch.Tensor().to(dtype=torch.long)
        
        cnt = 0
        
        for batch_index, batch in enumerate(tqdm(loader, desc='Batch')):
            batch = ou.to_device(batch, device=self._device)
            (_, inputs, labels) = batch
            
            embedding = model.backbone.forward_with_chosen_domains(x=inputs, domain_ids=domain_ids)
            # embedding_list.append(embedding.cpu().detach())
            embeddings_final = torch.cat([embeddings_final, embedding.cpu()], dim=0)
            labels_final = torch.cat([labels_final, labels.cpu()], dim=0)
            # labels_list.append(labels.cpu())
            
            if cnt > self.max_number_of_embeddings_in_memory:
                break
            
        return embeddings_final, labels_final
    
    # It will update the fc weights for the current domain with the prototypes of the classes.
    @torch.no_grad()
    def replace_fc_weight_with_prototypes(self, loader):
        if self.args.UMAP:
            return
        
        logging.info('Calculating the prototypes for the classifier ...')
        
        assert self.use_proxy_classifier
        
        embeddings, labels = self.obtain_embeddings_for_a_domain_or_domains(loader, domain_ids=self._cur_domain_id)
        model = self._network
        
        labels_remapped_from_zero = labels % self.class_num

        if self.classifier_type in [ClassifierType.Separate_Stochastic_Classifiers]:
            class_list_from_zero = np.unique(labels_remapped_from_zero)
            for class_index in class_list_from_zero:
                data_index = (labels_remapped_from_zero == class_index).nonzero().squeeze(-1)
                embedding = embeddings[data_index]
                proto = embedding.mean(0)
                model.classifiers_list[-1].mu.data[class_index, :] = proto
                
            if self.temporary_classifier_type in [ClassifierType.Single_Stochastic_Classifier]:
                model.classifiers_list[-1].sigma.data = self._network.proxy_fc.sigma.data
        else:
            if self.classifier_type in [ClassifierType.Separate_LinearLayers]:
                class_list_from_zero = np.unique(labels_remapped_from_zero)
                
                for class_index in class_list_from_zero:
                    data_index = (labels_remapped_from_zero == class_index).nonzero().squeeze(-1)
                    embedding = embeddings[data_index]
                    proto = embedding.mean(0)
                    model.classifiers_list[-1].layer.weight.data[class_index, :] = proto
            else:       # The default mode!
                class_list_from_zero = np.unique(labels_remapped_from_zero)
                
                for class_index in class_list_from_zero:
                    data_index = (labels_remapped_from_zero == class_index).nonzero().squeeze(-1)
                    embedding = embeddings[data_index]
                    proto = embedding.mean(0)
                    model.classifiers_list[-1].weight.data[class_index, :] = proto
    
    def _calculate_the_number_of_samples_to_generate(self, dataloader):
        num_samples_to_generate = min(len(dataloader.dataset), self.max_number_of_embeddings_in_memory // (len(self.chosen_layers_for_intermediate_domain_classifiers) * (self.total_sessions - 1)))
        
        return num_samples_to_generate
    
    def _divide_a_mini_batch(self, dataloader):
        logging.info('Finding the number of samples to be generated ...')
        num_samples_to_generate = self._calculate_the_number_of_samples_to_generate(dataloader)
        seen_domains = self._cur_domain_id + 1
        num_samples_to_choose_from_current_domain = dataloader.batch_size // seen_domains  # batch_size * 1 / (k+1)
        num_samples_to_choose_generated_embeddings = dataloader.batch_size - num_samples_to_choose_from_current_domain  # batch_size * k / (k+1)
        
        logging.info(f'    We use {num_samples_to_choose_generated_embeddings} old + {num_samples_to_choose_from_current_domain} new samples for each batch')
        
        return num_samples_to_generate, num_samples_to_choose_from_current_domain, num_samples_to_choose_generated_embeddings
    
    def _train_domain_classifier(self, dataloader):
        
        if self.args.UMAP:
            return
        
        logging.info("Training the domain classifier ...")
        
        num_epochs = self.epochs_domain_classifier_training
        
        num_samples_to_generate, num_samples_to_choose_from_current_domain, num_samples_to_choose_generated_embeddings = self._divide_a_mini_batch(dataloader)
        
        gc.collect()
        
        self.calculate_and_store_stats_for_current_domain(dataloader)
        
        gc.collect()
        
        # 1- We generate synthetic embeddings for every chosen auxiliary domain classifier for all previous domains.
        domain_labels_generated = None
        
        if self._cur_domain_id > 0 and self.cache_synthetic_embeddings:
            generated_embeddings_for_each_chosen_block, domain_labels_generated = self._generate_synthetic_embeddings(num_samples=num_samples_to_generate)
        
        gc.collect()
        
        if num_epochs == 0:
            return
        
        optimizer_domain_classifier, scheduler_domain_classifier = self._obtain_optimizer_and_scheduler_for_domain_classifier()
        
        for epoch in tqdm(range(num_epochs), desc='Epoch'):
            loss_accumulated_for_reporting = 0.
            correct, total = 0, 0
            
            for batch_index, batch in enumerate(dataloader):
                batch = ou.to_device(batch, device=self._device)
                (_, inputs, labels) = batch
                
                embeddings_dict_current_domain = self.forward_without_LoRAs(inputs)
                    
                embeddings_dict_current_domain = self.process_embeddings_dict(embeddings_dict_current_domain)
                domain_labels_current_domain = torch.ones_like(labels) * self._cur_domain_id
                
                loss = 0.0
                loss_ball_generator = 0
                
                if self._cur_domain_id > 0 and not self.cache_synthetic_embeddings:
                    generated_embeddings_for_each_chosen_block, domain_labels_generated = self._generate_synthetic_embeddings(num_samples=num_samples_to_choose_generated_embeddings)
                
                for block_id in self.chosen_layers_for_intermediate_domain_classifiers:
                    embeddings_current_domain_current_chosen_block = embeddings_dict_current_domain[block_id]
                    
                    # mix old and new samples after the second
                    if self._cur_domain_id > 0:
                        generated_embeddings_current_chosen_block = generated_embeddings_for_each_chosen_block[block_id]
                        
                        selected_indices_current_domain = torch.randperm(len(embeddings_current_domain_current_chosen_block))[:num_samples_to_choose_from_current_domain]
                        
                        embeddings_current_domain_selected = embeddings_current_domain_current_chosen_block[selected_indices_current_domain]
                        
                        domain_labels_current_domain_selected = domain_labels_current_domain[selected_indices_current_domain]
                        
                        if self.cache_synthetic_embeddings:
                            selected_indices_old_domains = torch.randperm(len(generated_embeddings_current_chosen_block))[:num_samples_to_choose_generated_embeddings]
                            
                            embeddings_old_domains_selected = generated_embeddings_current_chosen_block[selected_indices_old_domains].to(self._device)
                            
                            domain_labels_old_domains_selected = domain_labels_generated[selected_indices_old_domains].to(self._device)
                        else:
                            embeddings_old_domains_selected = generated_embeddings_current_chosen_block.to(self._device)
                            
                            domain_labels_old_domains_selected = domain_labels_generated.to(self._device)
                        
                        embeddings_seen_domains = torch.cat((
                            embeddings_current_domain_selected,
                            embeddings_old_domains_selected)
                        )
                        
                        domain_labels_seen_domains = torch.cat((
                            domain_labels_current_domain_selected,
                            domain_labels_old_domains_selected)
                        )
                    else:
                        embeddings_seen_domains = embeddings_current_domain_current_chosen_block
                        domain_labels_seen_domains = domain_labels_current_domain
                    
                    if self._cur_domain_id > 0 and self.use_transformation_module:
                        embeddings_seen_domains = self._network.transformation_modules[block_id].forward(embeddings_seen_domains)
                        
                        ball_centers = self.domain_classifier_data.centers[block_id]
                        
                        loss_ball_generator = ball_loss_fast(embeddings=embeddings_seen_domains, labels=domain_labels_seen_domains, ball_centers=ball_centers, margin=self.args.margin)
                    else:
                        embeddings_seen_domains = embeddings_seen_domains
                    
                    domain_logits = self._network.domain_classifier.forward(embeddings=embeddings_seen_domains, block_id=block_id)  # only support single GPU now
                    
                    loss_classification = F.cross_entropy(domain_logits, domain_labels_seen_domains)
                    
                    loss_current_layer = loss_classification + self.lambda_2 * loss_ball_generator
                    
                    loss = loss + loss_current_layer / len(self.chosen_layers_for_intermediate_domain_classifiers)
                
                loss_accumulated_for_reporting += loss.item()
                
                optimizer_domain_classifier.zero_grad()
                loss.backward()
                optimizer_domain_classifier.step()
                
                _, domain_preds = torch.max(domain_logits, dim=1)
                correct += domain_preds.eq(domain_labels_seen_domains.expand_as(domain_preds)).cpu().sum()
                total += len(domain_labels_seen_domains)
                
            scheduler_domain_classifier.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            logging.info(f"Domain classifier training: "
                        f"Task {self._cur_domain_id}, "
                        f"Epoch [{epoch + 1}/{self.epochs_domain_classifier_training}] "
                        f"lr {scheduler_domain_classifier.get_last_lr()[0]:.5f} "
                        f"Loss {loss_accumulated_for_reporting / len(dataloader):.4f}, "
                        f"Train_acc {train_acc:.2f}")
        
    @torch.no_grad()
    def calculate_and_store_stats_for_current_domain(self, dataloader):
        # Computing the statistics for the current domain for the future
        logging.info('==> Computing the statistics ...')
        
        embeddings_dict = self.obtain_embeddings_for_stats_calculations(dataloader)
        
        # Calculating the centers
        if self.generation_strategy in [GenerationStrategy.GMM_with_generator_loss]:
            for block_id, embeddings in embeddings_dict.items():
                means = embeddings.mean(0, keepdim=True).to(self._device)
                
                means = means.clone().detach()
                
                self.domain_classifier_data.centers[block_id] = torch.cat([self.domain_classifier_data.centers[block_id], means])
                
            logging.info(f'    Statistics are ready for task: {self._cur_domain_id}')
        
        # GMM
        if self._cur_domain_id < self.total_sessions - 1 and self.generation_strategy in [GenerationStrategy.GMM, GenerationStrategy.GMM_with_generator_loss]:
            logging.info('==> Compression ...')
            
            for block_id in tqdm(self.chosen_layers_for_intermediate_domain_classifiers, desc='Block'):
                embeddings = embeddings_dict[block_id]
            
                compression_to_be_saved = GMM_EM(features=embeddings.numpy(), n_components=self.n_components, max_iter=self.max_iter_for_GMM, tol=self.tol_for_GMM)
                self.domain_classifier_data.GMM_params_dict_of_lists[block_id].append(compression_to_be_saved)
                
                embeddings_dict[block_id] = None
                
                gc.collect()
            
            logging.info('==> Compression is done!')
    
    def _generate_synthetic_embeddings(self, num_samples: int):
        if num_samples > 1000:
            logging.info('=> Generating synthetic embeddings')
        
        # resample train set from old domains
        generated_embeddings_for_each_chosen_block = {}
        
        for block_id in tqdm(self.chosen_layers_for_intermediate_domain_classifiers, desc='Block'):
            generated_embeddings_list = []
            
            block_id = str(block_id)
            
            for domain_label in tqdm(range(self._cur_domain_id), desc='Domain'):
                compression = self.domain_classifier_data.GMM_params_dict_of_lists[block_id][domain_label]
                generated_embeddings = random_sampling(num_samples=num_samples, compression=compression, n_components=self.n_components)
                
                generated_embeddings_list.append(generated_embeddings)
            
            generated_embeddings_for_each_chosen_block[block_id] = torch.tensor(np.vstack(generated_embeddings_list), dtype=torch.float32)
    
        # The domain labels are the same for all layers
        domain_labels_generated = torch.repeat_interleave(torch.arange(self._cur_domain_id), repeats=num_samples)
        
        if self._cur_domain_id == self.total_sessions - 1:
            self.domain_classifier_data.GMM_params_dict_of_lists = None
            gc.collect()
        
        return generated_embeddings_for_each_chosen_block, domain_labels_generated
        
    def get_scheduler(self, optimizer, epoch):
        if self.args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.args.min_lr)
        elif self.args.scheduler == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args.init_milestones, gamma=self.args.lr_default_decay)
        elif self.args.scheduler == 'constant':
            scheduler = None

        return scheduler
    
    def _find_the_best_intermediate_domain_classifier(self, domain_logits_dict: dict):
        logits_tensor = torch.Tensor().to(device=self._device, dtype=torch.float32)
        
        for block_id in self.chosen_layers_for_intermediate_domain_classifiers:
            logits = domain_logits_dict[block_id]   # [batch_size, num_domains]
            logits_tensor = torch.cat([logits_tensor, logits.unsqueeze(1)], dim=1)
        # logits_tensor.shape = [batch_size, num_chosen_layers, num_domains]
        
        probabilities = logits_tensor.softmax(dim=-1)   # Its shape: [batch_size, num_chosen_layers, num_domains]
        
        confidence_scores_for_each_domain_classifier = probabilities.max(dim=-1)[0]    # Its shape: [batch_size, num_chosen_layers]
        
        max_confidence_block_indices = confidence_scores_for_each_domain_classifier.argmax(dim=-1)  # Its shape: [batch_size]
        
        # We prefer the first layer that meets the confidence threshold. If none of the classifiers meet the confidence threshold, we use the classifier with the highest confidence.
        
        mask_exceeding_threshold = confidence_scores_for_each_domain_classifier >= self.confidence_threshold  # [batch_size, num_chosen_layers]
        exceeding_threshold = mask_exceeding_threshold.float()     # Its shape: [batch_size, num_chosen_layers]
        
        mask_any_classifier_meets_threshold = exceeding_threshold.sum(dim=-1) > 0  # [batch_size]
        # Since all true values are equal, it returns the first index.
        first_classifier_with_sufficient_confidence = exceeding_threshold.argmax(dim=-1)
        
        block_ids_final = torch.where(mask_any_classifier_meets_threshold, first_classifier_with_sufficient_confidence, max_confidence_block_indices)
        
        batch_size = logits_tensor.shape[0]
        final_logits = logits_tensor[torch.arange(batch_size), block_ids_final]    # [batch_size, num_domains]
        
        return final_logits
    
    @torch.no_grad()
    def detect_domain_id(self, x: T):
        
        if self._cur_domain_id == 0:
            domain_ids_predicted = torch.tensor([0] * x.shape[0], device=self._device)
        else:
            embeddings_dict = self.forward_without_LoRAs(x)
                
            embeddings_dict = self.process_embeddings_dict(embeddings_dict=embeddings_dict)
            
            if self.use_transformation_module:
                for block_id in self.chosen_layers_for_intermediate_domain_classifiers:
                    embeddings_dict[block_id] = self._network.transformation_modules[block_id].forward(embeddings_dict[block_id])
            
            domain_logits_dict = self._network.domain_classifier.forward_dictionaries(embeddings_dict)
            
            domain_logits = self._find_the_best_intermediate_domain_classifier(domain_logits_dict=domain_logits_dict)
            
            domain_ids_predicted = torch.max(domain_logits[:, :self._cur_domain_id + 1], dim=1)[1]  # predicted domain ids
            
        return domain_ids_predicted

    def forward_without_LoRAs(self, x):
        embeddings_dict_current_domain = self._network.forward_without_LoRAs(x, block_ids_to_return=self.chosen_layers_for_intermediate_domain_classifiers)
        
        return embeddings_dict_current_domain
    
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
        predicted_labels_without_oracle, predicted_labels_with_oracle, true_labels, domain_classification_accuracy = self._eval_cnn(self.test_loader)
        
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
        
        for iter_num, batch in enumerate(loader):
            batch = ou.to_device(batch, self._device)
            _, inputs, labels = batch
            
            domain_ids_predicted = self.detect_domain_id(inputs)
            
            outputs_without_oracle = self._network.forward(inputs, test=True, domain_ids=domain_ids_predicted)
            logits_without_oracle = outputs_without_oracle['logits']
            
            predicted_labels_without_oracle = torch.topk(logits_without_oracle, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            
            predicted_labels_without_oracle_list.append(predicted_labels_without_oracle.cpu().numpy())
            
            if not is_core50:
                domain_ids_ground_truth = labels // self.class_num
                
                domain_classification_accuracy_calculator.update(domain_ids_predicted, domain_ids_ground_truth)
                
                # With oracle
                outputs_with_oracle = self._network.forward(inputs, test=True, domain_ids=domain_ids_ground_truth)
                
                logits_with_oracle = outputs_with_oracle['logits']
                
                # predicted_labels_with_oracle = logits_with_oracle.argmax(dim=1) % self.class_num
                predicted_labels_with_oracle = torch.topk(logits_with_oracle, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
                
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
        )  # [N, topk]
