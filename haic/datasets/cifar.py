import pickle
import os
from typing import Optional, Callable

import torch
from torchvision.datasets import VisionDataset

from PIL import Image

import numpy as np


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

class Cifar10h(VisionDataset):
    """CIFAR-10H is a dataset of soft labels reflecting human perceptual
    uncertainty for the 10000-image CIFAR-10 test set, first appearing in
    
        Joshua C. Peterson, Ruairidh M. Battleday, Thomas L. Griffiths, 
            & Olga Russakovsky (2019). Human uncertainty makes classification 
            more robust. In Proceedings of the IEEE International Conference 
            on Computer Vision.
    
    URL: https://github.com/jcpeterson/cifar-10h
    URL: https://github.com/jcpeterson/cifar-10h/raw/refs/heads/master/data/cifar10h-probs.npy
    
    TODO: Rely on builtin Cifar10(test=True), make this as a wrapper.
    TODO: Download.
    """
    
    url = 'https://github.com/jcpeterson/cifar-10h/raw/refs/heads/master/data/cifar10h-probs.npy'
    filename = 'cifar10h-probs.npy'
    
    cifar_folder = 'cifar-10-batches-py'
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names'
    }
    batch_name = 'test_batch'
    
    def __init__(self, root='data/cifar-10h',
                 human_model=None,
                 random_state=None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 include_human_labels: bool = True
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.meta_file = os.path.join(root, self.cifar_folder, self.meta['filename'])
        self.data_file = os.path.join(root, self.cifar_folder, self.batch_name)

        # Load data
        data = Cifar10h._unpickle(self.data_file)
        self.data = data['data'].reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1)) # to HWC
        self.targets = data['labels']

        meta = Cifar10h._unpickle(self.meta_file)
        self.label_names = meta[self.meta['key']]

        # Load human labels
        human_probs = np.load(os.path.join(root, self.filename))
        if human_model is None or human_model == 'distribution':
            self.human_model = DistributionModel()
        elif human_model == 'majority':
            self.human_model = MajorityModel()
        elif human_model == 'sample':
            self.human_model = SamplingModel(random_state)
        else:
            raise ValueError('Unknown value for human_model')
        
        self.human_model.build(human_probs)
        
        self.include_human_labels = include_human_labels

    def labels(self):
        return [x for x in self.label_names]
        
    @staticmethod
    def _unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        # Compatibility with torchvision.datasets.CIFAR10
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            img = self.target_transform(target)
        
        if self.include_human_labels:
            return img, self.human_model.get(idx), target
        else:
            return img, target
