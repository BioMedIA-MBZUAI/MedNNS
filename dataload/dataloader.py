
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from medmnist import INFO, Evaluator
from medmnist import DermaMNIST, PneumoniaMNIST, PathMNIST, RetinaMNIST, BreastMNIST , BloodMNIST, ChestMNIST, OCTMNIST, OrganCMNIST, OrganSMNIST, OrganAMNIST, TissueMNIST
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from torchvision import transforms
import numpy as np

# Mapping dataset names to their respective MedMNIST classes
data_class_mapping = {
    'PathMNIST': PathMNIST,# x
    'ChestMNIST': ChestMNIST,# 
    'DermaMNIST': DermaMNIST, # X
    'OCTMNIST': OCTMNIST,# x
    'PneumoniaMNIST': PneumoniaMNIST,# x
    'RetinaMNIST': RetinaMNIST,# 
    'BreastMNIST': BreastMNIST,# X
    'BloodMNIST': BloodMNIST,# X
    'TissueMNIST': TissueMNIST,# 
    'OrganAMNIST': OrganAMNIST,# o
    'OrganCMNIST': OrganCMNIST, # X
    'OrganSMNIST': OrganSMNIST # X
}

def get_dataloaders(dataset_name, batch_size=32, ignore_train=False):
    """
    Load MedMNIST dataset based on the dataset name and return dataloaders.

    Args:
        dataset_name (str): Name of the dataset to load.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_name not in data_class_mapping:
        raise ValueError(f"Invalid dataset name '{dataset_name}'. Must be one of {list(data_class_mapping.keys())}.")

    DatasetClass = data_class_mapping[dataset_name]

    # Define transforms for training and validation/test
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Random crop to 224x224
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ToTensor(),             # Convert image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),      # Resize to 224x224
        transforms.ToTensor(),             # Convert image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
    ])

    # Download and prepare data
    if not ignore_train:
        train_dataset = DatasetClass(split='train', transform=train_transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    else:
        train_loader = None

    val_dataset = DatasetClass(split='val', transform=val_test_transform, download=True)
    test_dataset = DatasetClass(split='test', transform=val_test_transform, download=True)

    # Dataloaders
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Data information
    data_info = {
        'num_classes': len(np.unique(val_dataset.labels)),
        'image_size': val_dataset[0][0].shape,  # Shape of the
        'name': dataset_name
    }

    return  train_loader,val_loader, test_loader, data_info