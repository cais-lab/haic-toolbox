"""Utility classes and functions to manipulate with datasets."""

import os
import torch


def iterate_batched(dataset, batch_size=None):
    """
    Iterate over a dataset in batches using a DataLoader.
    
    This generator function creates a PyTorch DataLoader to efficiently
    iterate through the dataset in batches of specified size. If no batch
    size is provided, it processes the entire dataset as a single batch.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to iterate over. Should be a PyTorch Dataset object
        or any compatible iterable.
    batch_size : int, optional
        The number of samples per batch. If None (default), uses the
        entire dataset as one batch.
    
    Yields
    ------
    batch : torch.Tensor or tuple of torch.Tensor
        A batch of data from the dataset. The exact structure depends on
        the dataset's __getitem__ method, but typically consists of
        (features, targets) tuples or single tensors.
    """
    
    if batch_size is None:
        batch_size = len(dataset)
    
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
    
    for batch in loader:
        yield batch

def predict(model, dataset, batch_size=None):
    """
    Obtain model predictions for a given dataset.
    
    This is a convenience function that provides basic batched prediction
    functionality. It assumes the first component of each batch contains
    the input features for the model. The function automatically handles
    model evaluation mode and gradient computation disabling.

    Note: The main purpose of the method is to reduce boilerplate code. 
    However, it may be limiting in some cases since it's not always clear 
    which components of the dataset should be fed to the model. By default, 
    it assumes the first component (which covers most common cases).

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model to use for predictions.
    dataset : torch.utils.data.Dataset
        The dataset to generate predictions for.
    batch_size : int, optional
        The number of samples per batch. If None, uses the entire dataset
        as a single batch.

    Returns
    -------
    torch.Tensor
        A tensor containing all model predictions concatenated along
        the first dimension (batch dimension).
    """
    
    model.eval()
    
    # Batch component to feed to the model
    pos = 0
    results = []
    
    for i, batch in enumerate(iterate_batched(dataset, batch_size=batch_size)):
        
        if i == 0:
            if not isinstance(batch, (tuple, list)):
                raise ValueError(
                    f"Batch must be tuple or list, got {type(batch)}"
                )
            if len(batch) not in (2, 3):
                raise ValueError("Batch must contain 2 or 3 elements")
            
        with torch.no_grad():
            results.append(model(batch[pos]))
    
    return torch.cat(results, 0)

def extract(dataset, selector: int, *, batch_size=None):
    """
    Extract a specific component from a dataset.
    
    This function processes a dataset and extracts a particular component
    (specified by selector) from each batch, then concatenates the results.
    It supports both simple tensor extraction and complex nested structures
    like lists or tuples of tensors.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to extract components from.
    selector : int
        The index position of the component to extract from each batch.
    batch_size : int, optional
        The number of samples per batch. If None, uses the entire dataset
        as a single batch.

    Returns
    -------
    torch.Tensor or list of torch.Tensor
        - If the selected component is a single tensor: returns a concatenated tensor
        - If the selected component is a list/tuple of tensors: returns a list
          of concatenated tensors, one for each original tensor in the structure

    Examples
    --------
    >>> # Extract features (position 0) from dataset
    >>> features = extract(dataset, selector=0, batch_size=64)
    >>> 
    >>> # Extract multi-expert labels (position 2, as list of tensors)
    >>> expert_labels = extract(dataset, selector=2, batch_size=64)
    >>> # Returns: [tensor_expert1, tensor_expert2, tensor_expert3]
    """
    results = []

    for batch in iterate_batched(dataset, batch_size=batch_size):
        item = batch[selector]

        if isinstance(item, (list, tuple)):
            if not results:
                results = [[] for _ in range(len(item))]
            for i, tensor in enumerate(item):
                results[i].append(tensor)
        else:
            results.append(item)

    if isinstance(results[0], list):
        return [torch.cat(r, dim=0) for r in results]
    else:
        return torch.cat(results, dim=0)

def save_dataset(dataset, filename, **kwargs):
    """
    Save a TensorDataset to disk using PyTorch's serialization.
    
    Parameters
    ----------
    dataset : torch.utils.data.TensorDataset
        The dataset to be saved. Must be a TensorDataset instance.
    filename : str or path-like
        The path where the dataset should be saved. Typically uses
        `.pt` or `.pth` extension for PyTorch files.    
    **kwargs : dict, optional
        Additional arguments passed to torch.save(), such as:
        - pickle_protocol: protocol version for pickle
        - _use_new_zipfile_serialization: use new zipfile format
        
    Raises
    ------
    ValueError
        If the input dataset is not a TensorDataset instance.
    IOError
        If the file cannot be written to the specified path.        
    """

    if not isinstance(dataset, torch.utils.data.TensorDataset):
        raise ValueError('Only TensorDataset instances are supported.')

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    torch.save(dataset.tensors, filename, **kwargs)

def load_dataset(filename, **kwargs):
    """
    Load a TensorDataset from disk that was saved using `save_dataset()`.
    
    This function reconstructs a TensorDataset from a file previously saved
    with `save_dataset()`. It automatically unpacks the saved tensors and
    creates a new TensorDataset instance with the original tensor structure.

    Parameters
    ----------
    filename : str or path-like
        The path to the saved dataset file. Typically uses `.pt` or `.pth`
        extension for PyTorch files.
    **kwargs : dict, optional
        Additional arguments passed to `torch.load()`, such as:
        - map_location: device to load tensors onto (e.g., 'cpu', 'cuda:0')
        - weights_only: if True, only load tensors (security feature)
        - pickle_module: custom pickle module for deserialization

    Returns
    -------
    torch.utils.data.TensorDataset
        The reconstructed dataset containing all original tensors.
    """
    try:
        # Load the tensor tuple from file
        loaded_data = torch.load(filename, **kwargs)
        
        # Validate that we have a tuple suitable for TensorDataset
        if not isinstance(loaded_data, tuple):
            raise TypeError(
                f"Expected tuple of tensors, got {type(loaded_data)}. "
                f"File may not have been saved with save_dataset()."
            )
        
        if len(loaded_data) == 0:
            raise ValueError("Loaded tuple is empty - cannot create TensorDataset")
            
        # Validate all elements are tensors
        for i, item in enumerate(loaded_data):
            if not torch.is_tensor(item):
                raise TypeError(
                    f"Element {i} is {type(item)}, expected torch.Tensor. "
                    f"File may be corrupted or in wrong format."
                )
        
        return torch.utils.data.TensorDataset(*loaded_data)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {filename}: {str(e)}") from e

class SyntheticExpertDataset(torch.utils.data.Dataset):
    
    def __init__(self, base_dataset, expert_model, batch_size=None):
        super().__init__()
        
        self.base_dataset = base_dataset
        self.expert_model = expert_model
        
        # Process the dataset
        self.expert_labels = []
        if not len(self.base_dataset):
            return
        
        # Only 'classical' (X, y) datasets are
        # supported
        assert len(self.base_dataset[0]) == 2
        
        Xs = []
        ys = []
        for X, y in iterate_batched(base_dataset, batch_size):
            Xs.append(X)
            ys.append(y)
        Xs = torch.cat(Xs, 0)
        ys = torch.cat(ys, 0)

        # NOTE: We respect the Numpy-based interface of models
        #    however, it might be reasonable to change that
        self.data = expert_model.make_labels(
            Xs.numpy(),
            ys.numpy()
        )
        
        # TODO: Try to mimic the underlying dataset types
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        X, y = self.base_dataset[idx]
        return X, self.data[idx], y     

class ExternalExpertDataset(torch.utils.data.Dataset):
    """Почти то же, но позволяет просто подгрузить метки из заданного массива."""
    pass

class ProjectDataset(torch.utils.data.Dataset):
    """
    A dataset wrapper that projects samples to specific components/indices.
    
    This class creates a view of an existing dataset that selectively returns
    only specified components from each sample. It handles both single-index
    and multi-index projections with comprehensive validation and error handling.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The original dataset to wrap. Each sample should support indexing
        (typically tuple, list, or mapping-like objects).
    idxs : int, list, or tuple
        Index or indices of the components to extract from each sample.
        - If a single integer: returns that component directly
        - If a list/tuple: returns a tuple of the specified components

    Raises
    ------
    ValueError
        During initialization if any specified index is invalid for the
        dataset samples (checked against the first sample).
    IndexError
        During item access if the indices are invalid for a specific sample.
    TypeError
        If the dataset samples don't support indexing.    
    """
    
    def __init__(self, dataset, idxs):
        super().__init__()
        self.wrappee = dataset

        # Convert single index to tuple for consistent handling
        if isinstance(idxs, int):
            self.idxs = (idxs,)
            self.single_component = True
        else:
            self.idxs = tuple(idxs)
            self.single_component = False

        # Validate indices on first sample (if dataset is non-empty)
        if len(dataset) > 0:
            sample = dataset[0]
            for idx in self.idxs:
                try:
                    _ = sample[idx]
                except (IndexError, KeyError, TypeError) as e:
                    raise ValueError(
                        f"Index {idx} is invalid for dataset samples. "
                        f"Sample type: {type(sample)}, sample length: {len(sample) if hasattr(sample, '__len__') else 'N/A'}"
                    ) from e    
    
    def __len__(self):
        return len(self.wrappee)
    
    def __getitem__(self, idx):
        item = self.wrappee[idx]
        
        try:
            if self.single_component:
                return item[self.idxs[0]]
            else:
                return tuple(item[x] for x in self.idxs)
        except (IndexError, KeyError) as e:
            raise IndexError(
                f"Failed to extract indices {self.idxs} from sample {idx}. "
                f"Sample type: {type(item)}, sample length: {len(item) if hasattr(item, '__len__') else 'N/A'}"
            ) from e        

class MultiExpertDataset(torch.utils.data.Dataset):
    """
    A dataset wrapper that aggregates multiple expert predictions with base data.
    
    This class extends a base dataset by adding multiple sets of expert predictions
    (labels) for each sample. It's designed for learning-to-defer scenarios where
    a model can choose to defer to one of multiple experts.

    Parameters
    ----------
    base_dataset : torch.utils.data.Dataset
        The base dataset containing input features and true labels. Expected to
        return tuples of (features, true_labels) when indexed.
    expert_labels_list : list of torch.Tensor or array-like
        List of expert prediction arrays, where each array contains predictions
        from one expert for all samples in the base_dataset. Each array should
        have the same length as the base_dataset.

    Raises
    ------
    ValueError
        If the length of any expert labels array doesn't match the base dataset length.
    IndexError
        If the base dataset doesn't return (features, labels) tuples.
    """
    
    def __init__(self, base_dataset, expert_labels_list):
        self.base_dataset = base_dataset  # ProjectDataset
        self.expert_labels_list = expert_labels_list
        
        # Validate that all expert arrays have correct length
        base_len = len(self.base_dataset)
        for i, expert_labels in enumerate(self.expert_labels_list):
            if len(expert_labels) != base_len:
                raise ValueError(
                    f"Expert {i} has {len(expert_labels)} predictions, "
                    f"but base dataset has {base_len} samples"
                )        

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns
        -------
        int
            The number of samples, same as the base dataset length.
        """    
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Get a sample with features, multiple expert predictions, and true label.
        
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.
            
        Returns
        -------
        tuple
            A tuple containing:
            - x : features from the base dataset
            - m_list : list of expert predictions for this sample
            - y : true label from the base dataset
            
        Raises
        ------
        IndexError
            If the base dataset doesn't return a (features, labels) tuple.
        """    
        x, y = self.base_dataset[idx]
        m_list = [m[idx] for m in self.expert_labels_list]
        return x, m_list, y