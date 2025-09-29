import torch
from Supernet_trainning import create_and_save_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Supernet on MedMNIST dataset with Sandwich Rule")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name from MedMNIST")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for training")
    parser.add_argument('--work_dir', type=str, default='./work_dir', help="Base directory for saving results")
    parser.add_argument('--config', type=str, required=True, help="Path to training configuration JSON file")
    args = parser.parse_args()

    create_and_save_model(args.dataset, args.device, args.work_dir, args.config)