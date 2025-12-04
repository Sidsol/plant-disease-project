"""
EfficientNet-B0 model for plant disease classification.

Supports progressive unfreezing for transfer learning.
"""

from typing import List

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 for plant disease classification with progressive unfreezing.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        dropout_rate: Dropout rate for classifier
    """
    
    def __init__(
        self,
        num_classes: int = 38,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
        # Store feature blocks for progressive unfreezing
        # EfficientNet-B0 has 9 blocks (0-8) in backbone.features
        self.feature_blocks = list(self.backbone.features.children())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all backbone layers (for initial training)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("Backbone frozen. Only classifier head will be trained.")
        
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        print("Entire backbone unfrozen.")
        
    def unfreeze_last_n_blocks(self, n: int = 3):
        """
        Unfreeze the last n blocks of the backbone.
        
        EfficientNet-B0 has 9 blocks (indices 0-8).
        Unfreezing last 3 blocks means blocks 6, 7, 8.
        
        Args:
            n: Number of blocks to unfreeze from the end
        """
        # First freeze everything
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        # Then unfreeze the last n blocks
        num_blocks = len(self.feature_blocks)
        start_idx = max(0, num_blocks - n)
        
        for i in range(start_idx, num_blocks):
            for param in self.feature_blocks[i].parameters():
                param.requires_grad = True
        
        print(f"Unfroze last {n} blocks (blocks {start_idx} to {num_blocks - 1}).")
        
    def get_trainable_params(self) -> tuple:
        """
        Get trainable parameters with different learning rates.
        
        Returns:
            Tuple of (backbone_params, classifier_params) lists
        """
        # Separate backbone and classifier parameters
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        return backbone_params, classifier_params
    
    def get_optimizer_param_groups(
        self,
        backbone_lr: float = 1e-4,
        classifier_lr: float = 1e-3
    ) -> List[dict]:
        """
        Get parameter groups with different learning rates for optimizer.
        
        Args:
            backbone_lr: Learning rate for backbone (lower)
            classifier_lr: Learning rate for classifier head (higher)
            
        Returns:
            List of parameter groups for optimizer
        """
        backbone_params, classifier_params = self.get_trainable_params()
        
        param_groups = []
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': backbone_lr,
                'name': 'backbone'
            })
            
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': classifier_lr,
                'name': 'classifier'
            })
        
        return param_groups
    
    def count_parameters(self) -> dict:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_pct': 100 * trainable / total
        }


def create_efficientnet_b0(
    num_classes: int = 38,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3
) -> EfficientNetB0:
    """
    Factory function to create EfficientNet-B0 model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone initially
        dropout_rate: Dropout rate for classifier
        
    Returns:
        Configured EfficientNet-B0 model
    """
    model = EfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    
    if freeze_backbone:
        model.freeze_backbone()
    
    return model
