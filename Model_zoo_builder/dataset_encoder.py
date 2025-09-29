import os
import random
import torch
import torchvision
from torchvision import transforms
from medmnist import DermaMNIST, PneumoniaMNIST, PathMNIST, RetinaMNIST, BreastMNIST, BloodMNIST, ChestMNIST, OCTMNIST, OrganCMNIST, OrganSMNIST, OrganAMNIST, TissueMNIST
import argparse
# Mapping dataset names to their respective MedMNIST classes
data_class_mapping = {
    'PathMNIST': PathMNIST,
    'ChestMNIST': ChestMNIST,
    'DermaMNIST': DermaMNIST,
    'OCTMNIST': OCTMNIST,
    'PneumoniaMNIST': PneumoniaMNIST,
    'RetinaMNIST': RetinaMNIST,
    'BreastMNIST': BreastMNIST,
    'BloodMNIST': BloodMNIST,
    'TissueMNIST': TissueMNIST,
    'OrganAMNIST': OrganAMNIST,
    'OrganCMNIST': OrganCMNIST,
    'OrganSMNIST': OrganSMNIST,
}

def get_test_dataset(dataset_name):
    """
    Downloads the test split for a given MedMNIST dataset using transforms
    that resize the image to 224x224, converts 1-channel images to 3 channels,
    and applies ImageNet normalization.
    """
    if dataset_name not in data_class_mapping:
        raise ValueError(f"Dataset {dataset_name} is not supported. Choose from {list(data_class_mapping.keys())}.")

    DatasetClass = data_class_mapping[dataset_name]
    
    # Define transform:
    # 1. Resize to 224x224.
    # 2. Convert to tensor.
    # 3. If the image has one channel, replicate to 3 channels.
    # 4. Normalize using ImageNet statistics.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        transforms.Normalize(mean=[0.5],
                             std=[0.5])
    ])
    
    # Download the test split of the dataset
    dataset = DatasetClass(split='test', transform=transform, download=True)
    return dataset

def sample_random_images(dataset, num_samples=64):
    """
    Randomly samples num_samples images from the given dataset.
    Assumes that each dataset item is a tuple (image, label) and returns a
    tensor of shape [num_samples, 3, 224, 224].
    """
    dataset_size = len(dataset)
    if dataset_size < num_samples:
        raise ValueError(f"Dataset only has {dataset_size} images, cannot sample {num_samples}.")
    
    indices = random.sample(range(dataset_size), num_samples)
    # Extract the images (ignore labels)
    images = [dataset[i][0] for i in indices]
    # Stack into a single tensor
    batch = torch.stack(images, dim=0)
    return batch

def get_dataset_encodings(num_samples=64):
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load a pretrained ResNet50 and remove the classification head.
    # The resulting model outputs 2048-dimensional features.
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove the final fully connected layer
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Dictionary to store encodings for each dataset
    dataset_encodings = {}

    # (Optional) Create a directory to store the output file if needed
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # For reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Loop over each dataset in the mapping
    for dataset_name in data_class_mapping.keys():
        print(f"Processing dataset: {dataset_name}")
        
        # Get the test dataset with the proper transforms
        dataset = get_test_dataset(dataset_name)
        
        # Sample 64 random images
        batch = sample_random_images(dataset, num_samples=64)
        batch = batch.to(device)

        # Encode images using ResNet50
        with torch.no_grad():
            encoding = model(batch)  # encoding shape: [64, 2048]
        
        # Move the encoding back to CPU and store in the dictionary
        dataset_encodings[dataset_name] = encoding.cpu()

    # Save the dictionary of encodings to a .pth file
    output_path = os.path.join(output_dir, "medmnist_resnet50_encodings.pth")
    torch.save(dataset_encodings, output_path)
    print(f"Encodings saved to {output_path}")

if __name__ == "__main__":
    # get args from CLI
    parser = argparse.ArgumentParser(description="Generate dataset encodings for MedMNIST datasets using Resnet50.")
    parser.add_argument('--num_samples', type=int, default=64, help='Number of random samples to draw from each dataset (default: 64)')
    args = parser.parse_args()
    
    get_dataset_encodings(num_samples=args.num_samples)