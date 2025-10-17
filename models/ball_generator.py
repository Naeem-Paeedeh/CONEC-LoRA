# Some code are modified from the FLOWER repository.

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor as T
import random
import numpy as np
import logging


def remap_the_labels_and_centers(
    labels: torch.Tensor,        # (B,) of dtype long, values in [0, C-1]
    ball_centers: torch.Tensor,  # (C, D), one center per class id)
) -> torch.Tensor:
    labels_unique_list = labels.unique().tolist()
    
    # num_classes = len(labels_unique_list)
    # num_samples = embeddings.shape[0]
    
    # Centers for remapped labels
    required_centers = torch.Tensor().to(ball_centers.device)

    labels_remapped = labels.clone()
          
    # Labels that remapped to start from zero.
    for i, lbl in enumerate(labels_unique_list):
        labels_remapped[labels_remapped == lbl] = i
        required_centers = torch.cat([required_centers, ball_centers[lbl].unsqueeze(0)], dim=0)
      
    return labels_remapped, required_centers


def ball_loss_simple(embeddings: T,
                     labels: T,
                     ball_centers: T,
                     margin: float
                     ) -> T:
    labels, ball_centers = remap_the_labels_and_centers(labels, ball_centers)
    # class_labels = list(set(labels.tolist()))
    class_labels = labels.unique().tolist()
    loss = 0.0
    
    for i in range(0, embeddings.shape[0]):
        # The center for the current class
        y_i = labels[i]
        center_i = ball_centers[y_i]
        # For every other class
        for y_j in range(min(class_labels), max(class_labels) + 1):
            if y_j == y_i:
                continue
            center_j = ball_centers[y_j]
            # Distance to its own center (prototype).
            di = torch.norm(embeddings[i] - center_i)
            # Distance to the center (prototype) of the other class.
            dj = torch.norm(embeddings[i] - center_j)
            dij = di - dj + margin
            dij = F.relu(dij)
            loss = loss + dij

    return loss


def ball_loss_fast(
    embeddings: torch.Tensor,    # (B, D)
    labels: torch.Tensor,        # (B,)
    ball_centers: torch.Tensor,  # (C, D), one center per domain label
    margin: float,
    reduction: str = "sum"
) -> torch.Tensor:
    device = embeddings.device
    
    labels_remapped, required_centers = remap_the_labels_and_centers(labels, ball_centers)
    num_classes = len(labels.unique().tolist())
    num_samples = embeddings.shape[0]
    
    # 1- Distances of the embeddings to their own centers.
    corresponding_ball_centers = required_centers[labels_remapped]
    dist_own_centers = torch.norm(embeddings - corresponding_ball_centers, dim=1)   # (num_samples,)

    # 2- Distances to all centers (for negatives)
    dist_all = torch.cdist(embeddings, required_centers, p=2)                         # (num_samples, num_classes)

    # 3- A mask for i != j
    idx = torch.arange(num_classes, device=device).unsqueeze(0)               # (1, num_classes)
    mask = idx.expand(num_samples, -1) != labels_remapped.unsqueeze(1)             # (num_samples, num_classes)

    # 4- Hinge loss
    dis_other_centers = dist_all[mask].view(num_samples, num_classes - 1)               # (num_samples, num_classes - 1)
    losses = F.relu(dist_own_centers.unsqueeze(1) - dis_other_centers + margin)  # (num_samples, num_classes- 1)
    
    # 5- Aggregation
    return losses.sum()
