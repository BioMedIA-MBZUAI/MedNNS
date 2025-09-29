from .dataset_encoder import get_dataset_encodings
from .model_encoder import model_encoder_main
import os
import torch

def aggregate_encodings():
    get_dataset_encodings(num_samples=64)

    for dataset in [
        'PathMNIST', 'ChestMNIST', 'DermaMNIST', 'OCTMNIST', 
        'PneumoniaMNIST', 'RetinaMNIST', 'OrganMNISTAxial', 
        'OrganMNISTCoronal', 'OrganMNISTSagittal', 'OrganSMNIST'
    ]:
        print(f"Processing model encodings for dataset: {dataset}")
        results = model_encoder_main(dataset, device='cuda:0', num_samples=1000, seed=42)
        # Save results to a file
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{dataset}_model_encodings.pth")
        torch.save(results, output_path)
        print(f"Model encodings saved to {output_path}")