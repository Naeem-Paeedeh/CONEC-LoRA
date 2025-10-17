import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy


class BaseLearner(object):
    def __init__(self, args):
        self._cur_domain_id = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._device = args.device[0]
        self._multiple_gpus = args.device
        self.args = args

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim
    
    def tsne(self, showcenters=False, Normalize=False):
        import umap
        import matplotlib.pyplot as plt
        print('now draw tsne results of extracted features.')
        tot_classes = self._total_classes
        test_dataset = self.data_manager.get_dataset(np.arange(0, tot_classes), source='test', mode='test')
        valloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        vectors, y_true = self._extract_vectors(valloader)
        if showcenters:
            fc_weight = self._network.fc.proj.cpu().detach().numpy()[:tot_classes]
            print(fc_weight.shape)
            vectors = np.vstack([vectors, fc_weight])
        
        if Normalize:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(vectors)
        
        if showcenters:
            clssscenters = embedding[-tot_classes:, :]
            centerlabels = np.arange(tot_classes)
            embedding = embedding[:-tot_classes, :]
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_true, s=20, cmap=plt.cm.get_cmap("tab20"))
        plt.legend(*scatter.legend_elements())
        if showcenters:
            plt.scatter(clssscenters[:, 0], clssscenters[:, 1], marker='*', s=50,
                        c=centerlabels, cmap=plt.cm.get_cmap("tab20"), edgecolors='black')

        plt.savefig(str(self.args.model_name) + str(tot_classes) + 'tsne.pdf')
        plt.close()

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_domain_id,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_domain_id))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T, y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true), decimals=2
        )

        return ret
    
    def eval_task(self):
        DIL_accuracy, DIL_accuracy_with_oracle, domain_classification_accuracy = self._eval_cnn(self.test_loader)

        return DIL_accuracy, DIL_accuracy_with_oracle, domain_classification_accuracy

    def incremental_train(self):
        pass

    def _train(self):
        pass
    
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network.forward(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
