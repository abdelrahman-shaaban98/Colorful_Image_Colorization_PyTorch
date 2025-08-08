import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def annealed_mean_softmax(logits, T=0.5):
    # logits: (B, 313, H, W)
    scaled = logits / T
    probs = F.softmax(scaled, dim=1)
    return probs  # Return soft probabilities over 313 bins


class RebalancedCrossEntropy(nn.Module):
    def __init__(self, classes_weights_path, device):
        super().__init__()
        self.class_weights = np.load(classes_weights_path)
        self.class_weights = torch.tensor(self.class_weights).float().to(device)

    def forward(self, pred, target):
        # pred: (B, 313, H, W)
        # target: (B, H, W) — each value in [0, 312]
        loss = F.cross_entropy(pred, target, weight=self.class_weights)
        return loss