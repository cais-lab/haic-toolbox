import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable

class DistributionModel:
    def build(self, value_probs):
        self.value_probs = value_probs
    
    def get(self, idx):
        return self.value_probs[idx]

class MajorityModel:
    def build(self, value_probs):
        self.values = np.argmax(value_probs, -1)
    
    def get(self, idx):
        return self.values[idx]

class SamplingModel:
    def __init__(self, random_state=None):
        self.rng = np.random.default_rng(seed=random_state)
    
    def build(self, value_probs):
        self.values = [
            self.rng.choice(value_probs.shape[1], 1, p=probs).item()
            for probs in value_probs
        ]
    
    def get(self, idx):
        return self.values[idx]

class GalaxyZoo(Dataset):
    def __init__(self, root='data/galaxy-zoo',
                 transform: Optional[Callable] = None,
                 human_model: Optional[str] = 'distribution',
                 include_human_labels: bool = True,
                 random_state: Optional[int] = None,
                 max_samples: Optional[int] = None):

        self.image_folder = os.path.join(root, 'images_training_rev1')
        csv_file = os.path.join(root, 'training_solutions_rev1.csv')
        self.transform = transform
        self.include_human_labels = include_human_labels

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            all_data = []
            for row in reader:
                image_id = row[0]
                probs = np.array([float(row[1]), float(row[2])])
                probs = np.clip(probs, 1e-3, 1.0)
                probs = probs / np.sum(probs)
                all_data.append((image_id, probs))
        
        if max_samples:
            all_data = all_data[:max_samples]

        self.image_ids, self.probs = zip(*all_data)
        self.probs = np.stack(self.probs)

        if human_model == 'distribution':
            self.human_model = DistributionModel()
        elif human_model == 'majority':
            self.human_model = MajorityModel()
        elif human_model == 'sample':
            self.human_model = SamplingModel(random_state)
        else:
            raise ValueError("Unknown human_model: choose from 'distribution', 'majority', 'sample'")

        self.human_model.build(self.probs)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_ids[idx] + '.jpg')
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        target = int(np.argmax(self.probs[idx]))

        if self.include_human_labels:
            human_label = self.human_model.get(idx)
            return img, human_label, target
        else:
            return img, target
