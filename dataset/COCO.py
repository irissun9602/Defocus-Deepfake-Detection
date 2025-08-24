import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class COCODataset(Dataset):
    def __init__(self, root_dir, max_files=100000, image_size=299):
        self.root_dir = root_dir
        self.image_size = image_size
        self.image_list = []
        self.max_files = max_files
        
        # Traverse subfolders to collect PNG/JPG files
        total_files = sum(len(files) for _, _, files in os.walk(root_dir))
        with tqdm(total=total_files, desc="Loading Images", unit="file") as pbar:
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.endswith((".png",".jpg")):
                        self.image_list.append(os.path.join(dirpath, filename))
                        if len(self.image_list) >= self.max_files:
                            break
                    pbar.update(1)
                if len(self.image_list) >= self.max_files:
                    break


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

        return image, torch.tensor(0, dtype=torch.long), img_path
