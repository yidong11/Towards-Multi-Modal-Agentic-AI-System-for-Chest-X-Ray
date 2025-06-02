# encoding: utf-8

"""
Generate predictions for MIMIC-CXR dataset
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import custom data loader and model
from read_data import ChestXrayDataSet
from model import DenseNet121, CLASS_NAMES, N_CLASSES, DEVICE

def load_model(model_path, device):
    """
    Load trained model from checkpoint
    """
    model = DenseNet121(N_CLASSES).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_predictions(model, test_loader, device):
    """
    Generate predictions for test dataset
    """
    # Prepare lists to store results
    image_paths = []
    pred_probs = []
    
    # Disable gradient computation
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            # Move data to device
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Convert to numpy
            batch_probs = outputs.cpu().numpy()
            
            # Get image paths for this batch
            batch_image_paths = test_loader.dataset.image_names[
                batch_idx * test_loader.batch_size : 
                (batch_idx + 1) * test_loader.batch_size
            ]
            
            # Collect results
            image_paths.extend(batch_image_paths)
            pred_probs.append(batch_probs)
    
    # Concatenate predictions
    pred_probs = np.concatenate(pred_probs, axis=0)
    
    return image_paths, pred_probs

def process_predictions(image_paths, pred_probs, labels_df):
    """
    Process predictions into the specified format
    """
    results_df = pd.DataFrame(columns=['subject_id', 'study_id', 'classification_result'])
    
    # Process each prediction
    for filename, probs in zip(image_paths, pred_probs):
        # Extract study_id using regex
        study_id_match = re.search(r's(\d+)', filename)
        if not study_id_match:
            continue
        
        study_id = study_id_match.group(1)
        
        match = labels_df[labels_df['study_id'] == int(study_id)]
        if len(match) == 0:
            continue
        
        subject_id = str(int(match.iloc[0]['subject_id']))
        
        # Create an ordered classification result
        ordered_result = []
        
        # Add pathologies in the specified order
        for i, pathology in enumerate(CLASS_NAMES):
            prob = probs[i]
            prob_str = f"{prob:.4f}"
            ordered_result.append((pathology, prob_str))
        
        # Add to results DataFrame
        new_row = pd.DataFrame({
            'subject_id': [subject_id],
            'study_id': [study_id],
            'classification_result': [str(ordered_result)]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    return results_df

def main():
    LABELS_CSV = '../mimic-cxr-2.0.0-chexpert.csv' 
    MODEL_PATH = 'model.pth.tar'
    IMAGE_DIR = '../mimic_cxr/images'
    TEST_CSV = 'inference.csv'
    OUTPUT_CSV = './cheXNet_inference.csv'
    
    # Print device information
    print(f"Using Device: {DEVICE}")
    
    # Load labels DataFrame
    labels_df = pd.read_csv(LABELS_CSV)
    
    # Data transforms for test set
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create test dataset
    test_dataset = ChestXrayDataSet(
        data_dir=IMAGE_DIR, 
        image_list_file=TEST_CSV, 
        transform=test_transform
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    # Load trained model
    print("Loading trained model...")
    model = load_model(MODEL_PATH, DEVICE)
    
    # Generate predictions
    print("Generating predictions...")
    image_paths, pred_probs = generate_predictions(model, test_loader, DEVICE)
    
    # Process predictions into desired format
    print("Processing predictions...")
    results_df = process_predictions(image_paths, pred_probs, labels_df)
    
    # Save to CSV
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to {OUTPUT_CSV}")
    
    # Print sample results
    print("\nSample Results:")
    print(results_df.head())

if __name__ == '__main__':
    main()

