import os
import random
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CiFakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        Args:
            samples (list of tuples): (image_path, label)
            transform (callable, optional): image preprocessing
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path 

def get_train_val_datasets(train_dir, transform=None, val_ratio=0.1, seed=42):
    real_dir = os.path.join(train_dir, 'REAL')
    fake_dir = os.path.join(train_dir, 'FAKE')

    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 9:1 split
    real_train, real_val = train_test_split(real_images, test_size=val_ratio, random_state=seed)
    fake_train, fake_val = train_test_split(fake_images, test_size=val_ratio, random_state=seed)

    train_samples = [(path, 0) for path in real_train] + [(path, 1) for path in fake_train]
    val_samples   = [(path, 0) for path in real_val]   + [(path, 1) for path in fake_val]

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return CiFakeDataset(train_samples, transform), CiFakeDataset(val_samples, transform)

def get_test_dataset(test_dir, transform=None):
    samples = []
    for label_name, label in [('REAL', 0), ('FAKE', 1)]:
        folder = os.path.join(test_dir, label_name)
        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append((os.path.join(folder, f), label))
    return CiFakeDataset(samples, transform)
