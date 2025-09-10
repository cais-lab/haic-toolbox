"""Utility functions for joint model outputs, where
classes and deferral decisions are placed into one tensor.

Typically, it is a tensor with K + E columns, where
K is the number of classes and E is the number of experts
(it is assumed that class columns go before expert columns).

Examples are L_{CE} and L_{OvA} methods.
"""

import torch

from haic.policy import Policy


def model_predictions(logits: torch.tensor, n_classes=None):
    """
    Extract model predictions from joint output logits.
    
    This function processes the joint output tensor containing both class
    probabilities and expert deferral logits, and extracts the model's
    class predictions by ignoring the expert columns and selecting the
    class with the highest probability.

    Parameters
    ----------
    logits : torch.Tensor
        Joint output tensor of shape (B, K + E), where:
        - B: batch size
        - K: number of classes (first K columns)
        - E: number of experts (last E columns)
        The tensor contains both class logits and expert deferral logits.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B,) containing the predicted class indices
        (argmax over the first K columns only).
    """
    if n_classes is None:
        n_classes = logits.shape[1] - 1
    
    return logits[:,:n_classes].max(-1).indices

def extract_policy(logits, n_classes=None):
    """
    Extract the deferral policy from joint model-expert output logits.
    
    This function processes the joint output tensor and determines the
    deferral policy by identifying when the model chooses to defer to
    an expert rather than making its own class prediction.

    Parameters
    ----------
    logits : torch.Tensor
        Joint output tensor of shape (B, K + E), where:
        - B: batch size
        - K: number of classes (first K columns)
        - E: number of experts (last E columns)
    n_classes : int, optional
        Number of classes (K). If None, inferred as logits.shape[1] - 1,
        assuming there is exactly 1 expert.

    Returns
    -------
    Policy
        A Policy object containing:
        - deferral decisions (binary tensor indicating when to defer)
        - number of executors (experts + model)
        - identifiers for human experts
    """

    # TODO: support other cases
    assert n_classes is None or n_classes == logits.shape[1] - 1
    
    if n_classes is None:
        n_classes = logits.shape[1] - 1
    
    p = logits.max(-1).indices
    return Policy((p >= n_classes).to(torch.int8),
                  n_executors=2,
                  human_executors=[1])

def get_joint_model_train_step(n_classes, n_executors):
    """
    Create a typical training step function for a model with joint output.
    """
    
    def do_step(model, batch, loss_fn, device):
        X, m, y = batch
        X, m, y = X.to(device), m.to(device), y.to(device)
        y_hat = model(X)
        assert y_hat.shape[1] == n_classes + n_executors
        l = loss_fn(y_hat[:, :n_classes], m, y_hat[:, n_classes:], y.to(torch.long))
        return l

    return do_step
