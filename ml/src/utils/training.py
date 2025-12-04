"""
Training utilities for plant disease classification.

Includes class weight calculation, early stopping, metrics, and TensorBoard helpers.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_class_weights(dataset: torch.utils.data.Dataset) -> torch.Tensor:
    """
    Calculate inverse frequency class weights for imbalanced datasets.
    
    Args:
        dataset: PyTorch dataset with targets attribute or iterable (image, label) pairs
        
    Returns:
        Tensor of class weights
    """
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        labels = [label for _, label in dataset]
    
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total_samples = sum(class_counts.values())
    
    # Inverse frequency weighting: weight = total / (num_classes * count)
    weights = torch.tensor([
        total_samples / (num_classes * class_counts[i])
        for i in range(num_classes)
    ], dtype=torch.float)
    
    return weights


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy/F1
        verbose: Print messages when stopping
    """
    
    def __init__(
        self, 
        patience: int = 5, 
        min_delta: float = 0.0, 
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self._current_epoch = 0
        
    def __call__(self, score: float, epoch: Optional[int] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
            epoch: Current epoch number (optional, auto-increments if not provided)
            
        Returns:
            True if should stop, False otherwise
        """
        if epoch is None:
            self._current_epoch += 1
            epoch = self._current_epoch
        else:
            self._current_epoch = epoch
            
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} epochs without improvement")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered. Best epoch: {self.best_epoch}, Best score: {self.best_score:.4f}")
                return True
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self._current_epoch = 0


class MetricsCalculator:
    """Calculate and store training metrics."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        
    def calculate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict:
        """
        Calculate classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics including per-class F1 scores
        """
        accuracy = (y_true == y_pred).mean()
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
        }
    
    # Alias for backward compatibility
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return self.calculate(y_true, y_pred)
    
    def get_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> str:
        """Get detailed classification report."""
        return classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (12, 10),
        normalize: bool = True
    ) -> plt.Figure:
        """
        Create confusion matrix plot.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            figsize: Figure size
            normalize: Normalize by row (true labels)
            
        Returns:
            Matplotlib figure
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)  # Handle division by zero
            
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=False,  # Too many classes for annotations
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.xticks(rotation=45, ha='right', fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()
        
        return fig


class TensorBoardLogger:
    """Helper class for TensorBoard logging."""
    
    def __init__(self, log_dir: Path, experiment_name: Optional[str] = None):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for logs (can include experiment name)
            experiment_name: Name of this experiment run (optional, appended to log_dir)
        """
        if experiment_name:
            self.log_dir = Path(log_dir) / experiment_name
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        self.writer.add_scalar(tag, value, step)
        
    def log_scalars(self, scalars: Dict[str, float], step: int):
        """Log multiple scalar values."""
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
            
    def log_learning_rate(self, lr: float, step: int):
        """Log learning rate."""
        self.writer.add_scalar('LearningRate', lr, step)
        
    def log_figure(self, tag: str, figure: plt.Figure, step: int):
        """Log matplotlib figure."""
        self.writer.add_figure(tag, figure, step)
        plt.close(figure)
        
    def log_model_graph(self, model: nn.Module, input_tensor: torch.Tensor):
        """Log model architecture graph."""
        self.writer.add_graph(model, input_tensor)
        
    def log_hyperparameters(self, hparams: Dict, metrics: Dict):
        """Log hyperparameters and final metrics."""
        self.writer.add_hparams(hparams, metrics)
        
    def close(self):
        """Close the writer."""
        self.writer.close()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    filepath: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    extra_info: Optional[Dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        best_metric: Best validation metric (e.g., accuracy)
        filepath: Path to save checkpoint
        scheduler: LR scheduler state (optional)
        extra_info: Additional info to save (e.g., class_names, config)
    """
    checkpoint = {
        'epoch': epoch,
        'best_metric': best_metric,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    if extra_info is not None:
        checkpoint.update(extra_info)
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load model to (optional, auto-detects if None)
        
    Returns:
        Dictionary with checkpoint info including 'epoch' and 'best_metric'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint
