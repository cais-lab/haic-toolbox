from typing import Union
from numbers import Integral

import torch

from haic.policy import Policy

def coverage(policy,
             group: bool=True) -> float:
    """Calculates coverage - ratio of items, assigned to the model.
    
    Works in several modes:
    - it can estimate coverage of each executor;
    - or group executors (to humans/machines)
    """
    p = policy.data.reshape(-1, 1)
    N = p.shape[0]
    E = policy.n_executors
    # Binary mask of the policy
    t = torch.zeros(N, E, dtype=torch.int8).scatter_(1,
        p.to(torch.int64),
        1)
    if group:
        g2 = policy.human_executors
        g1 = [x for x in range(E) if x not in g2]
        return (t[:, g1].sum() / t.sum()).item()
    else:
        return t.sum(0) / N

def accuracy(preds, gt) -> float:

    assert preds.shape == gt.shape
    
    return (preds == gt).to(torch.float32).mean().item()

def restricted_accuracy_bound(y_pred,
                              m_pred,
                              y_true,
                              coverage) -> float:
    """Maximal theoretical accuracy for given dataset with 
    the restricted coverage.

    Maximal theoretical accuracy can be achieved by the following
    optimal allocation procedure, taking into account coverage
    restriction:
        - assign to the expert as many items, correctly
          classified by the expert and incorrectly by the model,
          as possible;
        - assign to the model as many items, correctly
          classified by the model and incorrectly by the expert,
          as possible;
        - assign other items arbitrarily, as they are equally
          classified by both the expert and the model.
    Note, that such procedure might not exist, but *if* it
    existed, it would yield to the best possible coverage-accuracy
    curve.

    Parameters
    ----------
    y_pred : torch.tensor, np.ndarray
        Model predictions, (n_samples, ) or (n_samples, 1).
    m_pred : torch.tensor, np.ndarray
        Expert predictions, (n_samples, ) or (n_samples, 1).
    y_true : torch.tensor, np.ndarray
        True classes, (n_samples, ) or (n_samples, 1).
    coverage : Union[float, Integral]
        Coverage specification. It it is integral, it is
        the number of samples that must be assigned to the
        model, if it is a float number [0.0; 1.0] it is
        the respective ratio.

    Returns
    -------
    float
        Maximal theoretical accuracy value."""
    
    # Number of samples, correcly predicted by the expert and
    # incorrectly by the model
    n_notm_e = ((y_pred != y_true) & (m_pred == y_true)).sum()
    # Number of samples, correcly predicted by the model and
    # incorrectly by the expert
    n_m_note = ((y_pred == y_true) & (m_pred != y_true)).sum()
    # Number of samples, correcly predicted by both the model and
    # the expert
    n_m_e = ((y_pred == y_true) & (m_pred == y_true)).sum()
    n = len(y_pred)

    if isinstance(coverage, Integral):
        assert coverage >= 0 and coverage <= n
    else:
        assert coverage >= 0.0 and coverage <= 1.0
        coverage = int(coverage * n)
    
    return (min(n - coverage, n_notm_e) + 
            min(coverage, n_m_note) +
            n_m_e) / n

def optimal_coverage_accuracy_curve(y_pred: torch.tensor,
                                    m_pred: torch.tensor,
                                    y_true: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
    """Returns data, describing optimal coverage-accuracy curve
    for a given dataset.

    Calculations are based on the optimal sample distribution,
    described in the year 1 technical report.

    See also: restricted_accuracy_bound

    Parameters
    ----------
    y_pred: torch.tensor
        Model predictions, (N, 1) or (N, )
    m_pred: torch.tensor
        Human predictions, (N, 1) or (N, )
    y_true: torch.tensor
        True values, (N, 1) or (N, )
    Returns
    -------
    tuple(torch.tensor)
    Pair of tensors: the first is coverage, the second is
    respective accuracy. Coverage values are non increasing.
    """

    assert y_pred.shape == m_pred.shape
    assert m_pred.shape == y_true.shape
    assert y_pred.dim() == 1 or (y_pred.dim() == 2 and y_pred.shape[1] == 1)
    
    # Number of samples, correcly predicted by the expert and
    # incorrectly by the model
    n_notm_e = ((y_pred != y_true) & (m_pred == y_true)).sum()
    # Number of samples, correcly predicted by the model and
    # incorrectly by the expert
    n_m_note = ((y_pred == y_true) & (m_pred != y_true)).sum()
    # Number of samples, correcly predicted by both the model and
    # the expert
    n_m_e = ((y_pred == y_true) & (m_pred == y_true)).sum()
    n = len(y_pred)

    return (torch.tensor([n,
                          n - n_notm_e,
                          n_m_note,
                          0]) / n,
            torch.tensor([n_m_e + n_m_note,
                          n_m_note + n_m_e + n_notm_e,
                          n_m_note + n_m_e + n_notm_e,
                          n_m_e + n_notm_e]) / n
           )

def ac_auc(c: torch.tensor, a: torch.tensor) -> float:
    """AC-AUC metric.
    
    Area under provided coverage-accuracy curve.
    
    NOTE: coverage must be ordered, e.g., returned
    by coverage_accuracy_curve function.
    
    See also: haic.scoring.coverage_accuracy_curve
    
    Parameters
    ----------
    c: torch.tensor
        Coverage values (float in [0.0; 1.0]), (N, 1) or (N, )
    a: torch.tensor
        Accuracy values, (N, 1) or (N, )
    Returns
    -------
    float
    Metric value."""

    assert c.shape == a.shape
    assert c.dim() == 1 or (c.dim() == 2 and c.shape[1] == 1)
    assert len(c) > 1

    s = 0.0
    for i in range(len(c) - 1):
        s += (c[i + 1] - c[i]) * (a[i + 1] + a[i])
    return abs((s / 2).item())

def nac_auc(c: torch.tensor,
            a: torch.tensor,
            h_pred: torch.tensor,
            m_pred: torch.tensor,
            y_true: torch.tensor) -> float:
    """nAC-AUC metric for a given CA-curve and given
    dataset.
    
    This metric characterizes the whole CA-curve, describing
    how well samples are distributed between model and
    human executors. The closer CA-curve to the optimal
    one, the better distribution model leverages the information
    about strong sides of the executors.
    """
    co, ao = optimal_coverage_accuracy_curve(h_pred, m_pred, y_true)
    rand = ((ao[0] + ao[-1]) / 2).item()
    real = ac_auc(c, a)
    opt = ac_auc(co, ao)
    return (real - rand) / (opt - rand)
