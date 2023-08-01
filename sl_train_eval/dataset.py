import pandas as pd 
import os 
from PIL import Image 
from torch.utils.data import Dataset

class ImageLabelDataset(Dataset):
    def __init__(self, root, csv_path, transform):
        self.root = root
        df = pd.read_csv(csv_path)
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform
        self.num_classes = max(df["label"]) + 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])).convert("RGB"))
        label = self.labels[idx]
        return image, label
