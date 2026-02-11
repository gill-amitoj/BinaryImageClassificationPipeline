# ----------------------------------------------------------
# Intelligent Image Classification System
# Author: Amitoj Singh (CCID: amitoj3)
# ----------------------------------------------------------

import os
from typing import List, Tuple

import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import ImageFile
from pathlib import Path

# Skip obviously problematic files/folders at load time
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def _is_valid_file(path: str) -> bool:
    p = Path(path)
    # Ignore any file inside hidden or special folders (e.g., _corrupted)
    if any(part.startswith("_") or part.startswith(".") for part in p.parts):
        return False
    return p.suffix.lower() in ALLOWED_EXTS


class FilteredImageFolder(ImageFolder):
    """ImageFolder that ignores classes (subfolders) starting with '_' or '.'"""

    def find_classes(self, directory: str) -> Tuple[List[str], dict]:  # type: ignore[override]
        classes = [
            d.name
            for d in os.scandir(directory)
            if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
        ]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# Allow loading truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_data_loaders(train_dir, val_dir, batch_size=32, img_size=224):
    """Return PyTorch dataloaders for training and validation."""
    transform_train = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FilteredImageFolder(
        train_dir, transform=transform_train, is_valid_file=_is_valid_file, allow_empty=True
    )
    val_dataset = FilteredImageFolder(
        val_dir, transform=transform_val, is_valid_file=_is_valid_file, allow_empty=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(train_dataset.classes)

