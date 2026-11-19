"""
    Define the load of data and the preprocessing

    Key points:
    - preprocessing uses the official parameters of pretrained weight Mobile NetV3-Small (ridimension 256, central cut 224, normalization ImageNet)
    - The augmentation of data applied only to the training set
    - class Dataset read from CSV
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

# Obtain the official weights and the official preprocessing transformation of MobileNetV3-Small

weights = MobileNet_V3_Small_Weights.DEFAULT

# Ridimension the short side to 256 -> central cut 224 x 224
preprocess = weights.transforms()

# Transformation of training set
# the augmentation increment the variety of training data, preventing overfitting
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), # Random resize and cut to 224x224 and from 70% to 100%
    transforms.RandomHorizontalFlip(p=0.5), # Horizontal flip with 50% of probability to flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Slight variation of color: random modify brightnss, contrast and saturation
    transforms.ToTensor(), # Convert to Tensor (from PIL Image to numeric array, interval 0-1)
    # Normalization: uses mean and standard deviation of ImageNet
    # This values are stats of ImageNet dataset
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

val_transform = preprocess

# Personalized Dataset for Oxford-IIIT Pet
# Read image paths and labels from a CSV file
class PetDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df) # Total number of samples in the dataset

    # Return a sample based on the index
    def __getitem__(self, idx):
        row = self.df.iloc[idx] 
        img_path = os.path.join(self.image_root, row["image_path"]) # Relative path is "oxford-iiit-pet/images/name.jpg"

        # Convert image in RGB
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(batch_size=32, num_workers=2):
    # Create DataLoader for train, val and test
    # batch_size is the number of samples per batch
    # num_workers is the numbers of processes to load data

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    splits_dir = os.path.join(base_dir, "splits")
    data_root = os.path.join(base_dir, "data", "oxford-iiit-pet")

    # Creation of datasets
    train_dataset = PetDataset(
        csv_path=os.path.join(splits_dir, "train.csv"),
        image_root=data_root,
        transform=train_transform   
    )

    val_dataset = PetDataset(
        csv_path=os.path.join(splits_dir, "val.csv"),
        image_root=data_root,
        transform=val_transform   # validate without augmentation
    )

    test_dataset = PetDataset(
        csv_path=os.path.join(splits_dir, "test.csv"),
        image_root=data_root,
        transform=val_transform   # test without augmentation
    )

    # Creation of DataLoader
    # shuffle: True for train and False for the others to avoid that the model learn patterns linked to the order 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader