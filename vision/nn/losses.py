import torch
from torch import Tensor, nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor):
        """Computes the Dice loss.

        Args:
            logits (Tensor): C x H x W
            targets (Tensor): H x W
        """

        num_classes = logits.shape[0]
        dice = 0
        for c in range(num_classes):
            logits_c = logits[c]
            targets_c = targets == c
            intersection = torch.sum(logits_c * targets_c)
            union = torch.sum(logits_c) + torch.sum(targets_c) + self.eps
            dice += 2 * intersection / union
        return 1 - dice / num_classes


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor):
        """Computes the Focal loss.

        Args:
            logits (Tensor): C x H x W
            targets (Tensor): H x W
        """

        num_classes = logits.shape[0]
        focal = 0
        for c in range(num_classes):
            logits_c = logits[c]
            targets_c = targets == c
            intersection = torch.sum(logits_c * targets_c)
            union = torch.sum(logits_c) + torch.sum(targets_c)
            focal += (
                -1
                * self.alpha
                * (1 - intersection / union) ** self.gamma
                * torch.log(intersection / union)
            )
        return focal / num_classes
