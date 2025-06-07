# encoding: utf-8

"""
The main CheXNet model implementation for MIMIC-CXR dataset with verbose tracking.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Import custom data loader
from read_data import ChestXrayDataSet

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
IMAGE_DIR = './mimic_cxr/images'
TRAIN_CSV = './mimic_train.csv'
VALID_CSV = './mimic_valid.csv'
CKPT_PATH = 'model.pth.tar'

# Labels 
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Other', 'Pleural Effusion',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]
N_CLASSES = len(CLASS_NAMES)

class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def compute_metrics(labels, outputs):
    """
    Compute detailed metrics for multi-label classification
    """
    metrics = {}
    for i in range(labels.shape[1]):
        try:
            # AUROC
            auroc = roc_auc_score(labels[:, i], outputs[:, i])
            
            # Average Precision
            avg_precision = average_precision_score(labels[:, i], outputs[:, i])
            
            metrics[CLASS_NAMES[i]] = {
                'AUROC': auroc,
                'Average Precision': avg_precision
            }
        except ValueError:
            # Handle cases with only one class
            metrics[CLASS_NAMES[i]] = {
                'AUROC': np.nan,
                'Average Precision': np.nan
            }
    
    return metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Train the model with detailed progress tracking
    """
    print(f"{'='*50}")
    print(f"Starting Training on {DEVICE}")
    print(f"Total Epochs: {num_epochs}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"{'='*50}")

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Timing and tracking
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = len(train_loader)
        
        print(f"\nEPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*30}")
        
        # Progress tracking for training
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move data to device
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Zero parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
            
            # Print training progress
            if batch_idx % 10 == 0:
                print(f"Training Batch {batch_idx+1}/{train_batches}: Loss = {loss.item():.4f}")
        
        # Average training loss
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = len(val_loader)
        all_labels = []
        all_outputs = []
        
        print("\nValidation Phase:")
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Collect labels and outputs for metric computation
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
                
                # Print validation progress
                if batch_idx % 10 == 0:
                    print(f"Validation Batch {batch_idx+1}/{val_batches}: Loss = {loss.item():.4f}")
        
        # Concatenate all validation results
        all_labels = np.concatenate(all_labels, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        
        # Compute metrics
        val_metrics = compute_metrics(all_labels, all_outputs)
        
        # Compute average validation loss
        avg_val_loss = val_loss / val_batches
        
        # Epoch summary
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"\nEPOCH SUMMARY:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch Duration: {epoch_duration:.2f} seconds")
        
        # Print per-class metrics
        print("\nPer-Class Metrics:")
        for cls_name, metrics in val_metrics.items():
            print(f"{cls_name}:")
            print(f"  AUROC: {metrics['AUROC']:.4f}")
            print(f"  Avg Precision: {metrics['Average Precision']:.4f}")
        
        # Compute macro average
        macro_auroc = np.nanmean([m['AUROC'] for m in val_metrics.values()])
        macro_avg_precision = np.nanmean([m['Average Precision'] for m in val_metrics.values()])
        print(f"\nMacro AUROC: {macro_auroc:.4f}")
        print(f"Macro Avg Precision: {macro_avg_precision:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, CKPT_PATH)
            print(f"\nModel checkpoint saved. Best validation loss: {best_val_loss:.4f}")
        
        print(f"{'='*50}")
    
    return model

def main():
    # Ensure reproducibility
    torch.manual_seed(42)
    cudnn.benchmark = True

    # Print device information
    print(f"Using Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")

    # Data transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create datasets
    print("Preparing Training Dataset...")
    train_dataset = ChestXrayDataSet(
        data_dir=IMAGE_DIR, 
        image_list_file=TRAIN_CSV, 
        transform=train_transform
    )
    
    print("Preparing Validation Dataset...")
    val_dataset = ChestXrayDataSet(
        data_dir=IMAGE_DIR, 
        image_list_file=VALID_CSV, 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    # Print dataset information
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    
    # Initialize model
    model = DenseNet121(N_CLASSES).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        NUM_EPOCHS
    )
    
    # Final evaluation
    print("Training complete. Model saved to", CKPT_PATH)

if __name__ == '__main__':
    main()