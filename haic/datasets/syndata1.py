import numpy as np

import torch

class SyndataOne(torch.utils.data.Dataset):
    """Simple synthetic dataset.
    
    There are two classes and two features. The classes are well-separated in
    one area and not-so-well in other. However, in the area, when the classes
    are not so well separeted, expert labels are mostly true."""    

    def __init__(self, size: int,
                       dist: float=0.0,
                       mu: float=0.25,
                       random_state=None,
                       include_human_labels: bool = True):
        self.size = size
        self.mu = mu
        self.dist = dist
        self.random_state = random_state
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
        else:
            rng = np.random.default_rng()
        self.include_human_labels = include_human_labels

        delta = dist / 2
        # Class 0
        c1_cnt = self.size // 2
        c1x = rng.normal(0.5 - delta, self.mu, (c1_cnt,))
        c1y = rng.random(c1_cnt)
        c1x = c1x - (1 - c1y)*(0.5 - delta)
        X1 = np.stack([c1x, c1y], 1)
        
        # Class 1
        c2_cnt = self.size - c1_cnt
        c2x = rng.normal(0.5 + delta, self.mu, (c2_cnt,))
        c2y = rng.random(c2_cnt)
        c2x = c2x + (1 - c2y)*(0.5 - delta)
        X2 = np.stack([c2x, c2y], 1)
        
        self.X = np.concatenate([X1, X2]).astype(np.float32)
        self.y = np.concatenate([np.zeros(c1_cnt, dtype=np.int8),
                                 np.ones(c2_cnt, dtype=np.int8)])
        order = np.arange(self.size)
        rng.shuffle(order)
        self.X = self.X[order]
        self.y = self.y[order]
        
        correct = rng.random(self.size) < self.X[:, 1]
        self.m = np.where(correct, self.y, 1-self.y)
        
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.include_human_labels:
            return self.X[idx], self.m[idx], self.y[idx]
        else:
            return self.X[idx], self.y[idx]

