import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from collections import defaultdict
import glob 
import numpy as np
from torchvision import transforms

class FaceForensicsDataset(Dataset):
    def __init__(self, root_dir, fake_types=None, image_size=299, phase='train', transform=None):
        """
        FaceForensics++ dataset class
        - root_dir: root directory of FaceForensics++ dataset
        - fake_types: list of fake manipulation types (Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures)
        - image_size: image size (default: 224x224)
        - phase: choose from 'train', 'val', 'test' or specific subsets like 'real_train', 'fake_val', etc.
        """
        self.root_dir = root_dir
        self.fake_types = fake_types if fake_types else ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
        self.image_size = image_size
        self.phase = phase
        self.transform =transform
        
        # Load Real image paths
        self.real_dir = os.path.join(root_dir, "original_sequences", "youtube", "raw", "v1faces")
        self.real_images = self._load_file_list(self.real_dir, label=0)

        # Load Fake image paths (by manipulation type)
        self.fake_images = []
        for fake_type in self.fake_types:
            fake_dir = os.path.join(root_dir, "manipulated_sequences", fake_type, "raw", "v1faces")
            self.fake_images.extend(self._load_file_list(fake_dir, label=1))

        # Split Real and Fake separately into train/val/test
        self.real_train, self.real_val, self.real_test = self.split_by_folder(self.real_images)
        self.fake_train, self.fake_val, self.fake_test = self.split_by_folder(self.fake_images)

        # Keep separate lists for only Fake data
        self.fake_train_only = self.fake_train
        self.fake_val_only = self.fake_val
        self.fake_test_only = self.fake_test

        # Keep separate lists for only Real data
        self.real_train_only = self.real_train
        self.real_val_only = self.real_val
        self.real_test_only = self.real_test

        # Choose data according to the selected phase
        if self.phase == 'train':
            self.data = self.real_train + self.fake_train
        elif self.phase == 'val':
            self.data = self.real_val + self.fake_val
        elif self.phase == 'test':
            self.data = self.real_test + self.fake_test
        elif self.phase == 'real_train':
            self.data = self.real_train
        elif self.phase == 'real_val':
            self.data = self.real_val
        elif self.phase == 'real_test':
            self.data = self.real_test
        elif self.phase == 'fake_train':
            self.data = self.fake_train
        elif self.phase == 'fake_val':
            self.data = self.fake_val
        elif self.phase == 'fake_test':
            self.data = self.fake_test
        elif self.phase == 'real_train_only':
            self.data = self.real_train_only
        elif self.phase == 'real_val_only':
            self.data = self.real_val_only
        elif self.phase == 'real_test_only':
            self.data = self.real_test_only
        elif self.phase == 'fake_train_only':
            self.data = self.fake_train_only
        elif self.phase == 'fake_val_only':
            self.data = self.fake_val_only
        elif self.phase == 'fake_test_only':
            self.data = self.fake_test_only
                
        print(f"🔹 [{self.phase.upper()}] {fake_type if fake_type else 'Real'} image count: {len(self.data)}")

        #  Define transform
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_file_list(self, directory, label):
        if not os.path.exists(directory):
            print(f"Path does not exist: {directory}")
            return []

        image_list = []
        count = 0
        for root, _, files in os.walk(directory):
            # Pick only `.png` files
            images = sorted([os.path.join(root, file) for file in files if file.endswith(".png")])
            
            # Limit to maximum 32 images per folder
            images = images[:32]

            # Add (path, label) pairs
            for img in images:
                image_list.append((img, label))

        return image_list
    
    def split_by_folder(self, data_list):
        """
        Split data by folder in 7:1:2 ratio
        """
        folder_dict = defaultdict(list)
        
        # Group image paths by folder
        for img_path, label in data_list:
            folder_name = os.path.dirname(img_path)
            folder_dict[folder_name].append((img_path, label))
        
        # Sort folder names
        folders = sorted(folder_dict.keys())
        total_folders = len(folders)
        train_count = int(total_folders * 0.7)
        val_count = int(total_folders * 0.1)
        test_count = total_folders - (train_count + val_count)
        
        train_data, val_data, test_data = [], [], []

        # Split sequentially into 7:1:2
        for idx, folder in enumerate(folders):
            images = folder_dict[folder]
            if idx < train_count:
                train_data.extend(images)
            elif idx < train_count + val_count:
                val_data.extend(images)
            else:
                test_data.extend(images)
        
        return train_data, val_data, test_data
    
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        try:
            # Load image as PIL
            with Image.open(img_path) as img:
                image = img.convert("RGB")

             # Apply transform
            if self.transform:
                image = self.transform(image)  

        except Exception as e:
            print(f"Failed to load image: {img_path} - {e}")
            # Return dummy image if failed
            dummy = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return dummy, torch.tensor(-1), img_path

        return image, torch.tensor(label, dtype=torch.long), img_path