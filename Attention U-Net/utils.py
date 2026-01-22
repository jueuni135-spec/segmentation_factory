import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset

def get_train_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_data_loaders(batch_size=4):
    train_ds = SegmentationDataset('/content/split_dataset/train', transform=get_train_transforms())
    val_ds = SegmentationDataset('/content/split_dataset/val', transform=get_val_transforms())

    return {
        'training': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True),
        'validation': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    }