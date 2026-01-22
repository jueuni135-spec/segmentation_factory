import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calculate_miou(prediction, target, num_classes=4):
    pred = torch.argmax(prediction, dim=1)
    class_ious = []
    for cls in range(num_classes):
        p = (pred == cls)
        t = (target == cls)
        intersection = (p & t).sum().item()
        union = p.sum().item() + t.sum().item() - intersection
        if union == 0:
            class_ious.append(np.nan)
        else:
            class_ious.append((intersection + 1e-6) / (union + 1e-6))
    return np.nanmean(class_ious), class_ious

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3, weights=None):
        """
        alpha: False Negative(결함 놓침)에 대한 페널티 (0.7)
        beta: False Positive(오검출)에 대한 페널티 (0.3)
        gamma: Focal 계수
        weights: 클래스별 가중치 [1.0, 40.0, 70.0, 2.0]
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weights = weights

    def forward(self, logits, target):
        target = target.long()
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        tp = torch.sum(probs * target_oh, dims)
        fn = torch.sum(target_oh * (1 - probs), dims)
        fp = torch.sum((1 - target_oh) * probs, dims)
        
        # Tversky Index 계산
        tversky = (tp + 1e-6) / (tp + self.alpha * fn + self.beta * fp + 1e-6)
        
        # Focal 효과 적용
        focal_tversky = torch.pow((1 - tversky), self.gamma)
        
        # 사용자 지정 가중치 적용
        if self.weights is not None:
            focal_tversky = focal_tversky * self.weights
            
        return focal_tversky.mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes=4):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, target):
        # 🔥 RuntimeError 방지: target을 Long 타입으로 변환
        target = target.long()
        probs = F.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)
        dice_score = (2. * intersection / (cardinality + 1e-6)).mean()
        return 1 - dice_score

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        # 🔥 RuntimeError 방지: target을 Long 타입으로 변환
        target = target.long()
        ce_loss = F.cross_entropy(logits, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class HybridLoss(nn.Module):
    def __init__(self, alpha_weights):
        super(HybridLoss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=alpha_weights)

    def forward(self, logits, target):
        return 0.5 * self.dice(logits, target) + 0.5 * self.focal(logits, target)