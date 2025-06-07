# encoding: utf-8

"""
Read images and corresponding labels with extensive error checking.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import sys


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        print(f"Initializing dataset with:")
        print(f"  Data Directory: {data_dir}")
        print(f"  Image List File: {image_list_file}")

        # Validate input paths
        if not os.path.exists(data_dir):
            raise ValueError(f"Image directory does not exist: {data_dir}")
        
        if not os.path.exists(image_list_file):
            raise ValueError(f"Image list file does not exist: {image_list_file}")

        # Define the labels 
        self.labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Other', 'Pleural Effusion',
            'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]

        # Read the CSV file
        try:
            df = pd.read_csv(image_list_file)
            print(f"CSV loaded. Total rows: {len(df)}")
            print(f"CSV Columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            raise

        # Validate required columns
        required_columns = ['Image'] + self.labels
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")

        image_names = []
        label_values = []

        # Process each row in the dataframe
        for idx, row in df.iterrows():
            # Handle multiple images separated by semicolon
            img_paths = str(row['Image']).split(';')
            
            # Track if a valid image is found for this row
            path_found = False
            
            for img_path in img_paths:
                # Remove leading/trailing whitespace
                img_path = img_path.strip()
                
                # Construct full image path
                full_image_path = os.path.join(data_dir, img_path)
                
                # Check if image exists
                if os.path.exists(full_image_path):
                    image_names.append(full_image_path)
                    
                    # Extract label values
                    label = [row[label] for label in self.labels]
                    label_values.append(label)
                    path_found = True
                    break  # Use the first valid image path
            
            # Print warning if no valid image found for a row
            if not path_found:
                print(f"Warning: No valid image found for row {idx}: {row['Image']}")

        # Final validation
        if len(image_names) == 0:
            raise ValueError("No valid images found in the dataset. Check your image paths and directory.")

        print(f"Loaded {len(image_names)} valid images")

        self.image_names = image_names
        self.labels = label_values
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)