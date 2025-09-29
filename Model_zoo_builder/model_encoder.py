import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import random
import torch.nn.functional as F
import os
import numpy as np
import copy
import torchvision
from medmnist import INFO, Evaluator
from medmnist import (DermaMNIST, PneumoniaMNIST, PathMNIST, RetinaMNIST, 
                      BreastMNIST, BloodMNIST, ChestMNIST, OCTMNIST, 
                      OrganCMNIST, OrganSMNIST, OrganAMNIST, TissueMNIST)
import torchvision.transforms as transforms
from ofa.imagenet_classification.networks import ResNets
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer
)
from dataload import get_dataloaders

# Dataset mapping configuration
DATA_CLASS_MAPPING = {
    'BreastMNIST': BreastMNIST,
    'BloodMNIST': BloodMNIST,
    'PathMNIST': PathMNIST,
    'OrganAMNIST': OrganAMNIST,
    'OrganCMNIST': OrganCMNIST,
    'OrganSMNIST': OrganSMNIST,
    #'ChestMNIST': ChestMNIST,
    'DermaMNIST': DermaMNIST,
    'OCTMNIST': OCTMNIST,
    'PneumoniaMNIST': PneumoniaMNIST,
    #'RetinaMNIST': RetinaMNIST,
    'TissueMNIST': TissueMNIST,

}

DATA_PATH_MAPPING = {}
Main_dir = './work_dir/Supernets'
for ds in DATA_CLASS_MAPPING.keys():
    dir_path = os.path.join(Main_dir, ds,"super_net_complete_with_sandwich.pth")
    DATA_PATH_MAPPING[ds] = dir_path

print(DATA_PATH_MAPPING)





def sample_active_subnet(supernet, sample_config):
    """
    Sample an active subnet configuration from the supernet.
    
    Args:
        supernet: The supernet model
        sample_config (dict): Configuration for sampling
        
    Returns:
        dict: Architecture configuration
    """
    # Sample expand ratio
    expand_setting = []
    for block in supernet.blocks:
        expand_setting.append(random.choice(sample_config['e']))
    
    # Sample depth
    depth_list = sample_config['d'][0]
    depth_setting = [random.choice(depth_list)]
    for stage_id in range(len(ResNets.BASE_DEPTH_LIST)):
        depth_list = sample_config['d'][stage_id + 1]
        depth_setting.append(random.choice(depth_list))
    
    # Sample width multiplier
    width_mult_index = [random.choice(sample_config['w'][i]) 
                       for i in range(len(sample_config['w']))]
    width_mult_setting = [
        list(range(len(supernet.input_stem[0].out_channel_list)))[width_mult_index[0]],
        list(range(len(supernet.input_stem[2].out_channel_list)))[width_mult_index[1]],
    ]
    for stage_id, block_idx in enumerate(supernet.grouped_block_index):
        stage_first_block = supernet.blocks[block_idx[0]]
        width_mult_setting.append(
            list(range(len(stage_first_block.out_channel_list)))[width_mult_index[stage_id + 2]]
        )
    
    arch_config = {"d": depth_setting, "e": expand_setting, "w": width_mult_setting}
    supernet.set_active_subnet(**arch_config)
    return arch_config


def validate(val_loader, model, criterion, device):
    """
    Validate model on given data loader.
    
    Args:
        val_loader: Data loader for validation
        model: Model to validate
        criterion: Loss function
        device: Device to run on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Handle label format
            if labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)
            else:
                labels = labels.squeeze()
            
            # Handle single channel images
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = running_loss / len(val_loader.dataset)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def sample_fixed_subnets(supernet, num_samples=10, seed=42):
    """
    Sample a fixed set of subnets using a deterministic seed.
    
    Args:
        supernet: The supernet model
        num_samples (int): Number of subnets to sample
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of sampled configurations
    """
    random.seed(seed)
    sampled_configs = []
    sample_config = {
        'd': [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
        'e': [0.2, 0.25, 0.35],
        'w': [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0,1,2]]
    }
    
    for _ in range(num_samples):
        sampled_configs.append(sample_active_subnet(supernet, sample_config))
    
    return sampled_configs


def evaluate_subnets(supernet, sampled_configs, val_loader, test_loader, device, seed=42, dataset_name=None):
    """
    Evaluate sampled subnets and compute performance metrics.
    
    Args:
        supernet: The supernet model
        sampled_configs (list): List of subnet configurations
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        list: List of evaluation results
    """
    results = []
    criterion = torch.nn.CrossEntropyLoss()
    supernet.to(device)
    supernet.eval()
    counter = 0
    for config in tqdm(sampled_configs, desc="Evaluating sampled subnets"):
        supernet.set_active_subnet(**config)
        subnet = supernet.get_active_subnet()
        
        # Compute validation and test accuracy
        val_loss, val_acc = validate(val_loader, subnet, criterion, device)
        test_loss, test_acc = validate(test_loader, subnet, criterion, device)
        #functional_encoding = get_functional_encoding(subnet, num_samples=64, seed=seed, device=device).to("cpu")
        
        results.append({
            "subnet_id":counter,
            "subnet_config": {'w': config['w'], 'd': config['d'],'e': config['e']},
            "num_parameters": sum(p.numel() for p in subnet.parameters()),
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "val_loss": val_loss,
            "test_loss": test_loss,
            #"functional_encoding": functional_encoding,
            "dataset_name":dataset_name
        })
        counter += 1
    
    return results


def get_functional_encoding(model, num_samples=64, seed=42, device='cuda:0'):
    """
    Compute functional encoding of a model using embeddings of Gaussian noise.
    
    Args:
        model: Model to encode
        num_samples (int): Number of noise samples
        seed (int): Random seed
        
    Returns:
        torch.Tensor: Functional encoding
    """
    torch.manual_seed(seed)
    x = torch.randn(num_samples, 3, 224, 224)
    x = x.to(device)
    linear = copy.deepcopy(model.classifier)
    linear.to(device)
    model.classifier = torch.nn.Identity()
    embed = model(x)
    model.classifier = linear
    return embed

def load_and_evaluate_supernet(dataset_name, device='cuda:0', num_samples=1000, seed=42):

    """
    Load supernet and evaluate subnets for a given dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        work_dir (str): Working directory path
        device (str): Device to use
        num_samples (int): Number of subnets to sample
        
    Returns:
        list: Evaluation results
    """
    
    model_path = DATA_PATH_MAPPING.get(dataset_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Supernet not found at {model_path}")
    # Load data
    _, val_loader, test_loader, data_info = get_dataloaders(dataset_name=dataset_name,batch_size=128, ignore_train=True)
    
    # Load supernet
    supernet = torch.hub.load('mit-han-lab/once-for-all', 'ofa_supernet_resnet50', pretrained=True)
    n_classes = data_info['num_classes']
    print(f"Number of classes for {dataset_name}: {n_classes}")
    
    # Modify classifier for the dataset
    supernet.classifier = DynamicLinearLayer([supernet.classifier.linear.linear.in_features], n_classes)
    supernet = supernet.to(device)
    supernet.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    

    
    # Sample and evaluate subnets
    sampled_configs = sample_fixed_subnets(supernet, num_samples=num_samples, seed=seed)
    results = evaluate_subnets(supernet, sampled_configs, val_loader, test_loader, device,seed=seed, dataset_name=dataset_name)
    
    return results

def model_encoder_main(dataset_name, device='cuda:0', num_samples=1000, seed=42):
    print(f"Evaluating subnets for dataset: {dataset_name}")
    results = load_and_evaluate_supernet(dataset_name, device=device, num_samples=num_samples, seed=seed)
    # Save results
    path=DATA_PATH_MAPPING.get(dataset_name)
    # remove last part of path
    output_dir = os.path.dirname(path)
    save_file = os.path.join(output_dir, f"{dataset_name}_subnet_evaluation_results.pth")
    torch.save(results, save_file)
    print(f"Results saved to {save_file}")
    return results

