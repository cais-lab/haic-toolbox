"""Expert models.

Useful for simulating human labels of various expert proficiency.

TODO: Currently, only NumPy interface is supported. Should we also support 
      PyTorch?

"""
from typing import Optional
from abc import abstractmethod

import numpy as np

class UserModel:
    """Base class for all user models.
    
    A user model (probabilistically) generates labels for
    the specified samples. Some models may use feature values,
    some may ignore them and rely on classes only."""

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    @abstractmethod
    def make_labels(self, X, y):
        pass


class ConfusionMatrixUserModel(UserModel):
    """User model with the given confusion matrix."""

    def __init__(self, confusion_matrix,
                       random_state: Optional[int] = None):
          
        super().__init__(random_state=random_state)
        
        assert (len(confusion_matrix.shape) == 2) and \
               (confusion_matrix.shape[0] == confusion_matrix.shape[1])

        self.confusion_matrix = confusion_matrix
        
    def make_labels(self, _, y):

        # (N_SAMPLES, 1), (N_SAMPLES, )
        assert (len(y.shape) == 2 and y.shape[-1] == 1) or len(y.shape) == 1
        
        n_samples = y.shape[0]
        rnd_shape = y.shape + (1, )
        return np.sum(self.rng.random(rnd_shape) > 
                      np.cumsum(self.confusion_matrix[y], -1), 
                      -1)


class UniformExpertiseUserModel(ConfusionMatrixUserModel):
    """User model with uniform expertise over classes."""
    
    def __init__(self, n_classes: int,
                       correct_proba: float,
                       random_state: Optional[int] = None):

        self.n_classes = n_classes
        self.correct_proba = correct_proba

        cm = np.empty((n_classes, n_classes))
        cm.fill((1 - correct_proba) / (n_classes - 1))
        np.fill_diagonal(cm, correct_proba)

        super().__init__(cm, random_state=random_state)
        

class ClassBasedExpertiseUserModel(ConfusionMatrixUserModel):
    """A model of an expert who is more confident in certain classes."""

    def __init__(self, n_classes: int,
                       high_classes: list[int],
                       high_classes_proba: float,
                       low_classes_proba: float,
                       restrict_choices: bool = True,
                       random_state: Optional[int] = None):

        assert len(high_classes) >= 2
        assert n_classes - len(high_classes) >= 2

        self.n_classes = n_classes
        self.high_classes = high_classes
        self.high_classes_proba = high_classes_proba
        self.low_classes_proba = low_classes_proba
        self.restrict_choices = restrict_choices
        
        # Probability of a correct answer
        tp = np.zeros((n_classes,))
        tp.fill(low_classes_proba)
        tp[high_classes] = high_classes_proba
        
        # Fill the confusion matrix
        if restrict_choices:
            # Good classes
            gc = np.zeros(n_classes, dtype=bool)
            gc[high_classes] = 1

            good_mask = np.zeros((n_classes, n_classes), dtype=np.int8)
            good_mask[gc] = 1
            good_mask = good_mask & good_mask.T

            cm_g = np.tile(((1 - tp) / (len(high_classes) - 1)).reshape(-1, 1), (1, n_classes))

            bad_mask = np.zeros((n_classes, n_classes), dtype=np.int8)
            bad_mask[~gc] = 1
            bad_mask = bad_mask & bad_mask.T

            cm_b = np.tile(((1 - tp) / (n_classes - len(high_classes) - 1)).reshape(-1, 1), (1, n_classes))

            cm = cm_g * good_mask + cm_b * bad_mask
            np.fill_diagonal(cm, tp)
            
        else:

            cm = np.tile(((1 - tp) / (n_classes - 1)).reshape(-1, 1), (1, n_classes))
            np.fill_diagonal(cm, tp)
                       
        super().__init__(cm, random_state=random_state)
