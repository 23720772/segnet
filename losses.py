import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss','FocalDiceLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, input, target):
        # 计算Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        prob = torch.sigmoid(input)
        p_t = prob * target + (1 - prob) * (1 - target)
        focal_loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss

        focal_loss = focal_loss.mean()

        # 计算Dice Loss
        input_sigmoid = torch.sigmoid(input)
        num = target.size(0)
        input_flat = input_sigmoid.view(num, -1)
        target_flat = target.view(num, -1)
        intersection = (input_flat * target_flat).sum(1)
        dice_loss = 1 - ((2. * intersection + self.smooth) /
                         (input_flat.sum(1) + target_flat.sum(1) + self.smooth)).mean()

        return focal_loss + dice_loss


