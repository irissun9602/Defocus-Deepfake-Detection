import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class DiffusiondbDataset(Dataset):
    def __init__(self, root_dir, max_files=100000, image_size=224):
        self.root_dir = root_dir
        self.image_size = image_size  # define image_size
        self.image_list = []
        self.max_files = max_files
        
        #  Load only PNG files 
        with tqdm(total=len(os.listdir(root_dir)), desc="Loading Images", unit="file") as pbar:
            for entry in os.scandir(root_dir):
                if entry.is_file() and entry.name.endswith(".png"):
                    self.image_list.append(entry.path)
                pbar.update(1)
                if len(self.image_list) >= self.max_files:
                    break

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Failed to load image: {img_path} - {e}")
            return torch.zeros((3, self.image_size, self.image_size)), torch.tensor(-1), img_path

        return image, torch.tensor(1, dtype=torch.long), img_path
