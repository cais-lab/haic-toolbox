"""This module implements a set of tools to implement variety of 
methods, based on the idea of `scoring`. This idea states that
we can have some scoring function, so that the higher the score,
the more reasonable it is to assign the object to the model.

The most obvious implementation of such score is output probability
(softmax activation value), however, it is not the only one.

Positive effect of such scoring is that if we can establish
cnsistent scoring function, we can also control the coverage
by simple calibration (without retraining the model)."""

import warnings
import torch

from haic.policy import Policy, as_policy

def softmax_score(logits: torch.tensor, apply_softmax: bool = True):
    """Calculates score as max softmax.
    
    Parameters
    ----------
    logits: torch.tensor
        Logit values (typically model output) or output probabilities.
        Shape should be (B, K), where B is batch/dataset size and
        K is the number of classes.
    apply_softmax: bool
        Controls if softmax should be applied. Default is True.

    Returns
    -------
    Max probability values.
    """
    if apply_softmax:
        probs = torch.softmax(logits, -1)
    else:
        probs = logits
    return probs.max(-1).values

def score_threshold_policy(scores: torch.tensor,
                           thr: float) -> Policy:
    return Policy((scores < thr).to(torch.int8), 2, [1])    

def predict_score_threshold(scores: torch.tensor,
                            preds: torch.tensor,
                            m: torch.tensor,
                            thr: float):
    """Use more generic approach: create policy -> predict(policy).

    DEPRECATED!
    
    See: score_threshold_policy, predict.
    """
    return torch.where(scores > thr, preds, m)

def predict(policy: Policy,
            preds: torch.tensor,
            m: torch.tensor):
    assert policy.n_executors == 2
    assert len(policy.human_executors) == 1 and policy.human_executors[0] == 1
    return torch.where(policy.data == 0, preds, m)

def coverage_accuracy_curve(scores: torch.tensor,
                            y_pred: torch.tensor,
                            m: torch.tensor,
                            y_true: torch.tensor):
    """Calculates data for coverage-accuracy curve.
    
    Parameters
    ----------
    scores: torch.tensor
        Score values.
    y_pred: torch.tensor
        Model predictions.
    m: torch.tensor
        Human predictions.
    y_true: torch.tensor
        True values.
    
    Returns
    -------
    Two unidimensional tensors: coverage and respective
    accuracy values.
    
    """
    assert scores.dim() == 1 or (scores.dim() == 2 and scores.shape[1] == 1)
    assert y_pred.dim() == 1 or (y_pred.dim() == 2 and y_pred.shape[1] == 1)
    assert y_pred.shape == m.shape
    assert y_pred.shape == y_true.shape
    
    N = torch.numel(scores)
    
    coverage = torch.zeros(N + 1)
    accuracy = torch.zeros(N + 1)
    
    values, order = torch.sort(scores.view(-1))
    y_pred = y_pred[order]
    m = m[order]
    y_true = y_true[order]
    
    correct = (y_pred == y_true).sum()
    coverage[0] = 1.0
    accuracy[0] = correct / N
    for i in range(N):
        # ith position changes from model to human
        if y_pred[i] == y_true[i]:
            correct -= 1
        if m[i] == y_true[i]:
            correct += 1
            
        coverage[i+1] = (N - i - 1) / N
        accuracy[i+1] = correct / N
    
    return coverage, accuracy

def coverage_accuracy_curve_general(
    scores: torch.Tensor,
    y_pred: torch.Tensor,
    expert_predictions: torch.Tensor,
    y_true: torch.Tensor,
):
    """
    Coverage-accuracy кривая для произвольных делегирований (эксперт для каждого примера может быть разным).
    """
    warnings.warn("Deprecated. Use coverage_accuracy_curve", DeprecationWarning)
    
    scores = scores.view(-1)
    y_pred = y_pred.view(-1)
    expert_predictions = expert_predictions.view(-1)
    y_true = y_true.view(-1)

    assert scores.shape == y_pred.shape == expert_predictions.shape == y_true.shape
    N = scores.numel()

    coverage = torch.zeros(N + 1)
    accuracy = torch.zeros(N + 1)

    _, order = torch.sort(scores)
    y_pred = y_pred[order]
    expert_predictions = expert_predictions[order]
    y_true = y_true[order]

    correct = (y_pred == y_true).sum()
    coverage[0] = 1.0
    accuracy[0] = correct / N

    for i in range(N):
        if y_pred[i] == y_true[i]:
            correct -= 1
        if expert_predictions[i] == y_true[i]:
            correct += 1

        coverage[i + 1] = (N - i - 1) / N
        accuracy[i + 1] = correct / N

    return coverage, accuracy