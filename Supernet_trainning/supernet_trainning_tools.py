import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from medmnist import INFO, Evaluator
from medmnist import DermaMNIST, PneumoniaMNIST, PathMNIST, RetinaMNIST, BreastMNIST , BloodMNIST, ChestMNIST, OCTMNIST, OrganCMNIST, OrganSMNIST, OrganAMNIST, TissueMNIST
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from ofa.imagenet_classification.networks import ResNets
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import json
from dataload import get_dataloaders



def network_size(dynamic_net):
    """
    Computes the number of parameters in the dynamic network.
    """
    return sum(p.numel() for p in dynamic_net.parameters())

def get_extreme_configs(sample_config):
    """
    Generate the largest and smallest subnet configurations.
    
    Args:
        sample_config: Configuration with available choices
    
    Returns:
        max_config, min_config: Largest and smallest subnet configurations
    """
    # Maximum configuration (largest subnet)
    max_expand = [max(sample_config['e']) for _ in range(len(sample_config['e']))]
    max_depth = [max(sample_config['d'][i]) for i in range(len(sample_config['d']))]
    max_width = [max(sample_config['w'][i]) for i in range(len(sample_config['w']))]
    
    max_config = {"d": max_depth, "e": max_expand, "w": max_width}
    
    # Minimum configuration (smallest subnet)
    min_expand = [min(sample_config['e']) for _ in range(len(sample_config['e']))]
    min_depth = [min(sample_config['d'][i]) for i in range(len(sample_config['d']))]
    min_width = [min(sample_config['w'][i]) for i in range(len(sample_config['w']))]
    
    min_config = {"d": min_depth, "e": min_expand, "w": min_width}
    
    return max_config, min_config

def sample_active_subnet(supernet, sample_config):
    """
    Sample and set an active subnet configuration.
    """
    # sample expand ratio
    if isinstance(sample_config['e'][0], list):
        expand_setting = [random.choice(sample_config['e'][i]) for i in range(len(sample_config['e']))]
    else:
        expand_setting = [random.choice(sample_config['e']) for _ in range(len(supernet.blocks))]

    # sample depth
    depth_setting = []
    for stage_id in range(len(sample_config['d'])):
        depth_list = sample_config['d'][stage_id]
        depth_setting.append(random.choice(depth_list))

    # sample width_mult
    width_mult_index = [random.choice(sample_config['w'][i]) for i in range(len(sample_config['w']))]
    width_mult_setting = [
        list(range(len(supernet.input_stem[0].out_channel_list)))[width_mult_index[0]],
        list(range(len(supernet.input_stem[2].out_channel_list)))[width_mult_index[1]],
    ]
    for stage_id, block_idx in enumerate(supernet.grouped_block_index):
        stage_first_block = supernet.blocks[block_idx[0]]
        width_mult_setting.append(
            list(range(len(stage_first_block.out_channel_list)))[width_mult_index[stage_id+2]]
        )

    arch_config = {"d": depth_setting, "e": expand_setting, "w": width_mult_setting}
    supernet.set_active_subnet(**arch_config)
    return arch_config

def train_one_epoch_sandwich_rule(train_loader, optimizer, criterion, dynamic_net, epoch, 
                                 sampling_config, dynamic_batch_size=2, device='cuda:0'):
    """
    Trains using the sandwich rule: largest, smallest, and random subnets.
    
    Args:
        train_loader: DataLoader for training
        optimizer: Optimizer
        criterion: Loss function
        dynamic_net: Supernet model
        epoch: Current epoch
        sampling_config: Configuration for sampling subnets
        dynamic_batch_size: Number of random subnets to sample between extremes
        device: Training device
    
    Returns:
        Average loss and accuracy
    """
    dynamic_net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Get extreme configurations
    max_config, min_config = get_extreme_configs(sampling_config)

    with tqdm(train_loader, desc=f"Sandwich Training Epoch #{epoch+1}") as t:
        for batch_idx, (images, labels) in enumerate(t):
            images, labels = images.to(device), labels.to(device)
            
            # Handle label formatting
            if labels.size(0) != 1:
                if labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)
                else:
                    labels = labels.squeeze()
            else:
                labels = labels[0]
                
            # Handle single channel images
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)

            optimizer.zero_grad()
            loss_of_subnets = []

            # 1. Train largest subnet
            dynamic_net.set_active_subnet(**max_config)
            outputs = dynamic_net(images)
            loss = criterion(outputs, labels)
            loss_of_subnets.append(loss)
            loss.backward()
            
            # Update metrics
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # 2. Train smallest subnet
            dynamic_net.set_active_subnet(**min_config)
            outputs = dynamic_net(images)
            loss = criterion(outputs, labels)
            loss_of_subnets.append(loss)
            loss.backward()
            
            # Update metrics
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # 3. Train random subnets
            for i in range(dynamic_batch_size):
                random.seed(epoch * len(train_loader) + batch_idx + i)
                arch_config = sample_active_subnet(dynamic_net, sampling_config)
                
                outputs = dynamic_net(images)
                loss = criterion(outputs, labels)
                loss_of_subnets.append(loss)
                loss.backward()
                
                # Update metrics
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            # Step optimizer after all subnets
            optimizer.step()

            # Average loss
            avg_loss = torch.stack(loss_of_subnets).mean().item()
            running_loss += avg_loss * images.size(0)

            # Update progress bar
            t.set_postfix(
                loss=avg_loss,
                accuracy=100.0 * correct / total,
            )

    avg_epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = 100.0 * correct / total
    return avg_epoch_loss, accuracy

def train_one_epoch_with_sampling(train_loader, optimizer, criterion, dynamic_net, epoch, 
                                dynamic_batch_size=4, sampling_config=None, device='cuda:0'):
    """
    Trains the OFA net for one epoch by sampling multiple subnets per batch.
    """
    dynamic_net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, desc=f"Training Epoch #{epoch+1}") as t:
        for batch_idx, (images, labels) in enumerate(t):
            images, labels = images.to(device), labels.to(device)
            
            if labels.size(0)!=1:
                if labels.size(1)>1:
                    labels=torch.argmax(labels, dim=1)
                else:
                    labels=labels.squeeze()
            
            else:
                labels=labels[0]
            if images.size(1)==1:
                images=images.repeat(1,3,1,1)

            # Reset gradients
            optimizer.zero_grad()

            # Accumulate loss for multiple subnets
            loss_of_subnets = []
            for _ in range(dynamic_batch_size):
                # Sample a random subnet and set it as active
                random.seed(epoch * len(train_loader) + batch_idx + _)
                sample_active_subnet(dynamic_net, sample_config=sampling_config)

                # Forward pass
                outputs = dynamic_net(images)
                loss = criterion(outputs, labels)
                loss_of_subnets.append(loss)

                # Backward pass for this subnet
                loss.backward()
                

                # Metrics
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            # Step the optimizer after accumulating gradients for all subnets
            optimizer.step()

            # Average the loss over all sampled subnets
            avg_loss = torch.stack(loss_of_subnets).mean().item()
            running_loss += avg_loss * images.size(0)

            # Update progress bar
            t.set_postfix(
                loss=avg_loss,
                accuracy=100.0 * correct / total,
            )

    avg_epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = 100.0 * correct / total
    return avg_epoch_loss, accuracy

def train_one_epoch(train_loader, optimizer, criterion, dynamic_net, epoch, device):
    """
    Trains the OFA net for one epoch using the full supernet.
    """
    dynamic_net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loss_list=[]
    with tqdm(train_loader, desc=f"Training Epoch #{epoch+1}") as t:
        for batch_idx, (images, labels) in enumerate(t):
            images, labels = images.to(device), labels.to(device)
            
            if labels.size(0)!=1:
                if labels.size(1)>1:
                    labels=torch.argmax(labels, dim=1)
                else:
                    labels=labels.squeeze()
            
            else:
                labels=labels[0]
            if images.size(1)==1:
                images=images.repeat(1,3,1,1)

            optimizer.zero_grad()

            # Forward pass
            outputs=F.softmax(dynamic_net(images))
            loss = criterion(outputs, labels)

            # Metrics
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()
            loss_list.append(loss)

            # Average the loss
            avg_loss = torch.stack(loss_list).mean().item()
            running_loss += avg_loss * images.size(0)

            # Update progress bar
            t.set_postfix(loss=avg_loss, accuracy=100.0 * correct / total)

    avg_epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = 100.0 * correct / total
    return avg_epoch_loss, accuracy

def validate(val_loader, criterion, dynamic_net, device):
    """
    Validates the supernet.
    """
    dynamic_net.eval()

    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            if labels.size(1)>1:
                labels=torch.argmax(labels, dim=1)
            else:
                labels=labels.squeeze()        
            if images.size(1)==1:
                images=images.repeat(1,3,1,1)
                
            outputs = dynamic_net(images)
            loss = criterion(outputs, labels)

            # Accumulate metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total

    mean_loss = avg_loss
    mean_accuracy = accuracy
    valid_log='Validation Loss: %.4f, Accuracy: %.2f' % (mean_loss, mean_accuracy)
    return mean_loss, mean_accuracy, valid_log

def validate_all_subnets(val_loader, criterion, dynamic_net, sampling_config=None, device='cuda:0'):
    """
    Validates multiple sampled subnets.
    """
    dynamic_net.eval()

    valid_log=''

    with torch.no_grad():
        for i in range(20):
            arch_config=sample_active_subnet(dynamic_net,sampling_config)
            subnet=dynamic_net.get_active_subnet()
            loss_list=[]
            accuracy_list=[]
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).squeeze()
                    if images.shape[1] == 1:
                        images = images.repeat(1, 3, 1, 1)
                    outputs = subnet(images)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            avg_loss = running_loss / total
            avg_acc= 100.0 * correct / total
            loss_list.append(avg_loss)
            accuracy_list.append(avg_acc)
            size=network_size(subnet)
            valid_log += f"Subnet {arch_config}, Size: {size} -> Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%\n"
    return loss_list,accuracy_list,valid_log

def load_training_config(config_path):
    """
    Load training configuration from JSON file.
    
    Args:
        config_path (str): Path to the JSON configuration file
        
    Returns:
        dict: Training configuration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def progressive_shrinking_training(super_net, train_loader, val_loader, test_loader, 
                                 training_config, device='cuda:0'):
    """
    Execute the complete progressive shrinking training pipeline based on configuration.
    
    Args:
        super_net: The supernet model
        train_loader, val_loader, test_loader: Data loaders
        training_config: Training configuration loaded from JSON
        device: Training device
        
    Returns:
        dict: Training results
    """
    results = {}
    
    # Stage 1: Full supernet training
    stage_config = training_config['stages']['full_supernet']
    print(f"Stage 1: {stage_config['description']}")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(super_net.parameters(), lr=stage_config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_config['epochs'])

    train_losses, train_accuracies, val_accuracies = [], [], []
    
    for epoch in range(stage_config['epochs']):
        avg_loss, accuracy = train_one_epoch(
            train_loader, optimizer, criterion, super_net, epoch, device=device
        )
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        print(f"Epoch [{epoch + 1}/{stage_config['epochs']}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if epoch % 10 == 0:
            val_loss, val_accuracy, valid_log = validate(val_loader, criterion, super_net, device=device)
            val_accuracies.append(val_accuracy)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        scheduler.step()
        print(f' Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

    # Test after full training
    test_loss, test_accuracy, test_log = validate(test_loader, criterion, super_net, device=device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    results['full_supernet'] = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }

    # Stage 2: Progressive depth shrinking
    stage_config = training_config['stages']['depth_shrinking']
    print(f"Stage 2: {stage_config['description']}")
    
    sampling_config = stage_config['sampling_config'].copy()
    
    for i, layer_config in enumerate(stage_config['layer_progression']):
        print(f'Training depth: {layer_config["description"]}')
        
        # Update sampling config
        for layer_idx in layer_config['layers']:
            sampling_config['d'][layer_idx] = layer_config['values']
        
        # Setup optimizer and criterion
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=layer_config.get('label_smoothing', 0.0))
        optimizer = torch.optim.SGD(super_net.parameters(), lr=layer_config['lr'], momentum=0.9)
        
        # Train for specified epochs
        for epoch in range(layer_config['epochs']):
            avg_loss, acc = train_one_epoch_with_sampling(
                train_loader, optimizer, criterion, super_net, epoch, 
                dynamic_batch_size=layer_config['dynamic_batch_size'], 
                sampling_config=sampling_config, device=device
            )

        
        # Validate
        val_loss, val_accuracy, valid_log = validate_all_subnets(
            test_loader, criterion, super_net, sampling_config=sampling_config, device=device
        )
        print(valid_log)
        
        results[f'depth_stage_{i}'] = {
            'train_loss': avg_loss,
            'train_accuracy': acc,
            'test_loss': val_loss,
            'test_accuracy': val_accuracy,
            'valid_log': valid_log
        }

    # Stage 3: Progressive width shrinking
    stage_config = training_config['stages']['width_shrinking']
    print(f"Stage 3: {stage_config['description']}")
    
    for i, layer_config in enumerate(stage_config['layer_progression']):
        print(f'Training width: {layer_config["description"]}')
        
        # Update sampling config
        for layer_idx in layer_config['layers']:
            sampling_config['w'][layer_idx] = layer_config['values']
        
        # Setup optimizer and criterion
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=layer_config.get('label_smoothing', 0.0))
        optimizer = torch.optim.SGD(super_net.parameters(), lr=layer_config['lr'], momentum=0.9)
        
        # Train for specified epochs
        for epoch in range(layer_config['epochs']):
            avg_loss, acc = train_one_epoch_with_sampling(
                train_loader, optimizer, criterion, super_net, epoch, 
                dynamic_batch_size=layer_config['dynamic_batch_size'], 
                sampling_config=sampling_config, device=device
            )
        
        # Validate
        val_loss, val_accuracy, valid_log = validate_all_subnets(
            test_loader, criterion, super_net, sampling_config=sampling_config, device=device
        )
        print(valid_log)
        
        results[f'width_stage_{i}'] = {
            'train_loss': avg_loss,
            'train_accuracy': acc,
            'test_loss': val_loss,
            'test_accuracy': val_accuracy,
            'valid_log': valid_log
        }

    # Stage 4: Sandwich rule training (NEW)
    stage_config = training_config['stages']['sandwich_rule']
    print(f"Stage 4: {stage_config['description']}")
    
    # Final sampling config with all dimensions enabled
    final_sampling_config = stage_config['final_sampling_config']
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=stage_config.get('label_smoothing', 0.1))
    optimizer = torch.optim.SGD(super_net.parameters(), lr=stage_config['lr'], momentum=0.9)
    
    sandwich_losses, sandwich_accuracies = [], []
    
    for epoch in range(stage_config['epochs']):
        avg_loss, acc = train_one_epoch_sandwich_rule(
            train_loader, optimizer, criterion, super_net, epoch,
            sampling_config=final_sampling_config,
            dynamic_batch_size=stage_config['dynamic_batch_size'],
            device=device
        )
        
        sandwich_losses.append(avg_loss)
        sandwich_accuracies.append(acc)
        
        print(f"Sandwich Epoch [{epoch + 1}/{stage_config['epochs']}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
        
        # Validate every few epochs
        if epoch % 5 == 0:
            val_loss, val_accuracy, valid_log = validate_all_subnets(
                test_loader, criterion, super_net, sampling_config=final_sampling_config, device=device
            )
            print(f"Sandwich Validation - Loss: {val_loss[0]:.4f}, Accuracy: {val_accuracy[0]:.2f}%")

    # Final validation
    final_val_loss, final_val_accuracy, final_valid_log = validate_all_subnets(
        test_loader, criterion, super_net, sampling_config=final_sampling_config, device=device
    )
    
    results['sandwich_rule'] = {
        'train_losses': sandwich_losses,
        'train_accuracies': sandwich_accuracies,
        'final_test_loss': final_val_loss,
        'final_test_accuracy': final_val_accuracy,
        'final_valid_log': final_valid_log
    }
    
    print("=== SANDWICH RULE FINAL VALIDATION ===")
    print(final_valid_log)
    
    return results

def create_and_save_model(dataset_name, device, work_dir, config_path):
    """
    Train and save the supernet model using configuration-based training.
    """
    # Load training configuration
    training_config = load_training_config(config_path)
    
    # Ensure directories exist
    dataset_dir = os.path.join(work_dir, "Supernets", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader, data_info = get_dataloaders(
        dataset_name, batch_size=training_config['batch_size']
    )

    
    # Load supernet
    super_net = torch.hub.load('mit-han-lab/once-for-all', 'ofa_supernet_resnet50', pretrained=True)
    n_classes = data_info['num_classes']

    # Modify classifier for the dataset
    super_net.classifier = DynamicLinearLayer([super_net.classifier.linear.linear.in_features], n_classes)
    super_net = super_net.to(device)

    # Execute progressive shrinking training
    results = progressive_shrinking_training(
        super_net, train_loader, val_loader, test_loader, training_config, device
    )

    # Save final results
    final_results = {
        'state_dict': super_net.state_dict(),
        'training_results': results,
        'config': training_config
    }
    
    model_path = os.path.join(dataset_dir, "super_net_complete_with_sandwich.pth")
    torch.save(final_results, model_path)
    print(f"Complete model saved at {model_path}")
