"""
PyTorch Dataset and DataLoader utilities for plant disease classification.

Provides data loading with augmentation and ImageNet normalization.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get training data transforms with augmentation.
    
    Args:
        image_size: Target image size (default 224 for EfficientNet)
        
    Returns:
        Compose transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Compose transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_datasets(
    data_dir: Path,
    image_size: int = 224
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    """
    Create train, validation, and test datasets.
    
    Args:
        data_dir: Path to processed data directory containing train/val/test folders
        image_size: Target image size
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = Path(data_dir)
    
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    
    # Verify directories exist
    for split_dir in [train_dir, val_dir, test_dir]:
        if not split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {split_dir}")
    
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms(image_size)
    )
    
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_val_transforms(image_size)
    )
    
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_val_transforms(image_size)
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for stable batch norm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_class_names(dataset: datasets.ImageFolder) -> list:
    """Get class names from dataset."""
    return dataset.classes


def get_num_classes(dataset: datasets.ImageFolder) -> int:
    """Get number of classes from dataset."""
    return len(dataset.classes)


def get_dataset_stats(dataset: datasets.ImageFolder) -> Dict:
    """
    Get dataset statistics.
    
    Args:
        dataset: ImageFolder dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    from collections import Counter
    
    class_counts = Counter(dataset.targets)
    
    return {
        'num_samples': len(dataset),
        'num_classes': len(dataset.classes),
        'class_names': dataset.classes,
        'class_counts': {
            dataset.classes[idx]: count 
            for idx, count in sorted(class_counts.items())
        },
        'min_class_count': min(class_counts.values()),
        'max_class_count': max(class_counts.values()),
        'imbalance_ratio': max(class_counts.values()) / min(class_counts.values())
    }


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    return tensor * std + mean
