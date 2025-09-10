import warnings
import math

import torch
import torch.nn.functional as F


LOG2_OF_E = math.log2(math.e)

def softmax_parametrization_loss(hx, m, rx, target):
    """Softmax parametrization loss from Mozannar and Sontag.

    See also: Mozannar H., Sontag D. Consistent estimators for learning to 
        defer to an expert // 37th International Conference on Machine Learning, 
        ICML 2020. 2020. Vol. PartF16814. pp. 7033–7044.

    Implementation of the learning-to-defer loss function based on Equation 3
    from the original paper. Note that the appendix of the paper
    (https://arxiv.org/pdf/2006.01862) suggests an alternative formulation
    implemented in the `l1_ce_loss` function.

    Current implementation assumptions:
    - Classification cost c(i) is 0 for correct predictions and 1 for errors
    - Binary cost structure (0/1) for simplicity

    Limitations and future improvements:
    TODO: Support configurable cost matrices for asymmetric classification costs
    TODO: Add support for single-sample processing (non-batch mode)

    Parameters
    ----------
    hx : torch.Tensor
        Model class logits (before softmax), shape (batch_size, num_classes)
    m : torch.Tensor
        Human expert predictions, integer class indices [0, num_classes-1],
        shape (batch_size,)
    rx : torch.Tensor
        Delegation function outputs (real-valued scores), shape (batch_size,)
    target : torch.Tensor
        Ground truth labels, integer class indices [0, num_classes-1],
        shape (batch_size,)        
        
    """
    warnings.warn("Deprecated. Use l_ce_loss", DeprecationWarning)

    # The original implementation uses log2(softmax), however,
    # separate application of log and softmax can be numerically
    # unstable. Therefore, we use log_softmax with the 
    # following transformation from natural logarithm to log_2
    lsm = F.log_softmax(torch.cat([hx, rx.view(-1, 1)], -1), -1) * LOG2_OF_E
    
    # Classification cost (c)
    c = torch.ones_like(hx).scatter_(1, target.reshape(-1, 1), 0.0)
    # Add human classification cost
    c = torch.cat([c, (m != target).reshape(-1, 1)], -1)

    return torch.mean( 
        torch.sum(-(torch.max(c, -1, keepdim=True).values - c) * lsm, 
                  -1) 
    )

def l_ce_loss(hx, m, rx, target, alpha=1.0):
    """
    Loss function from Mozannar, Sontag, also referred to as sotmax
    parametrization loss.

    See also: Mozannar H., Sontag D. Consistent estimators for learning to 
        defer to an expert // 37th International Conference on Machine
        Learning, ICML 2020. 2020. Vol. PartF16814. pp. 7033–7044.

    This implementation follows the eq. 7 (in the ICML version of the
    paper), Appendix A (in the ArXiv version) and the accompanying repository 
    (https://github.com/clinicalml/learn-to-defer).
    In particular, it introduces parameter `alpha`, which purpose
    is to re-weight examples where the expert is correct to discourage the
    learner of fitting them and instead focus on examples where the expert
    makes a mistake.
    
    See also: https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf
    See also: https://arxiv.org/pdf/2006.01862
    See also: https://github.com/clinicalml/learn-to-defer/blob/master/cifar/cifar10h_defer.ipynb
    See also: `softmax_parametrization_loss`.

    Limitations and future improvements:
    TODO: Add support for single-sample processing (non-batch mode)

    Parameters
    ----------
    hx: torch.tensor
        Logits for classes, (B, K)
    m: torch.tensor
        Human expert predictions, integers in {0, ..., K-1}, (B, )
    rx: torch.tensor
        Rejector output, real (logit), (B, )
    target : torch.tensor
        Target class, integer in {0, ..., K-1}, (B, )
    alpha: float
        Parameter, controlling weight of the examples where the expert is
        correct, alpha >= 0
    """
    # For compatibility with the paper variable naming 
    outputs = torch.cat([hx, rx.view(-1,1)], 1)   # model outputs
    expert = (m == target).to(torch.float32)      # expert agreement labels
    batch_size, k_classes = hx.shape              # cardinality of target
    
    # Note: Original implementation uses softmax + log2, 
    # we use log_softmax for numerical stability
    lsm = F.log_softmax(outputs, dim=1) * LOG2_OF_E
    loss = -expert * lsm[range(batch_size) , k_classes]  \
           -(alpha * expert + (1 - expert)) * lsm[range(batch_size), target]
    return torch.sum(loss) / batch_size

def l_ce_loss_multi(hx, m_list, rx_list, target, alpha=1.0):
    outputs = torch.cat([hx] + [r.view(-1, 1) for r in rx_list], dim=1)
    batch_size, num_classes = hx.shape
    lsm = F.log_softmax(outputs, dim=1) * LOG2_OF_E

    expert_loss = 0.0
    for i, m in enumerate(m_list):
        expert_correct = (m == target).float()
        expert_wrong = 1.0 - expert_correct
        m_i = alpha * expert_wrong + expert_correct
        idx = num_classes + i
        expert_loss += -m_i * lsm[range(batch_size), idx]

    expert_correct = (m_list[0] == target).float()
    expert_wrong = 1.0 - expert_correct
    m2 = alpha * expert_wrong + expert_correct
    class_loss = -m2 * lsm[range(batch_size), target]

    total_loss = class_loss + expert_loss
    return torch.sum(total_loss) / batch_size

def logloss(x):
    return torch.log2(1 + torch.exp(-x))

def ova_loss(hx, m, rx, target, phi=logloss):
    """
    One-vs-All surrogate loss implementation based on:
    R. Verma, E. Nalisnick Calibrated Learning to Defer with One-vs-All Classifiers //
    Proceedings of the 39 th International Conference on Machine
    Learning, Baltimore, Maryland, USA, PMLR 162, 2022
    (https://arxiv.org/pdf/2202.03673)

    Limitations and future improvements:
    TODO: Add support for single-sample processing (non-batch mode)
    TODO: Account for alpha coefficient, introduced in the original
          implementation
 
    See also the authors' original implementation: https://github.com/rajevv/OvA-L2D
    
    Parameters
    ----------
    hx : torch.Tensor
        Class logit values, shape (batch_size, num_classes)
    m : torch.Tensor
        Human expert predictions, integer class indices [0, num_classes-1],
        shape (batch_size,)
    rx : torch.Tensor
        Delegation function outputs (real-valued scores), shape (batch_size,)
    target : torch.Tensor
        Ground truth labels, integer class indices [0, num_classes-1],
        shape (batch_size,)    
    
    Returns
    -------
    torch.Tensor
        Computed loss value, scalar
    """
    batch_size, n_classes = hx.shape

    assert m.shape == (batch_size, ) or m.shape == (batch_size, 1)
    assert rx.shape == (batch_size, ) or rx.shape == (batch_size, 1)
    assert target.shape == (batch_size, ) or target.shape == (batch_size, 1)

    m = m.view(-1)
    rx = rx.view(-1)
    target = target.view(-1)

    t1 = phi(hx[range(batch_size), target])
    #t2 = (torch.ones_like(hx).scatter_(1, target, 0) * phi(-hx)).sum(-1)
    t2 = phi(-hx[range(batch_size), :]).sum(-1) - \
         phi(-hx[range(batch_size), target])
    t3 = phi(-rx[range(batch_size)])
    t4 = ((m == target) * 
          (phi(rx[range(batch_size)]) - phi(-rx[range(batch_size)])))

    return (t1 + t2 + t3 + t4).mean()

def ova_loss_multi(hx, m_list, rx_list, target, phi=logloss):

    batch_size, num_classes = hx.shape
    num_experts = len(rx_list)

    target = target.view(-1)
    t1 = phi(hx[range(batch_size), target])
    t2 = phi(-hx).sum(-1) - phi(-hx[range(batch_size), target])

    t3 = 0
    for rx in rx_list:
        t3 += phi(-rx.view(-1))

    t4 = 0
    for i, (rx, m) in enumerate(zip(rx_list, m_list)):
        m = m.view(-1)
        expert_correct = (m == target).float()
        t4 += expert_correct * (phi(rx.view(-1)) - phi(-rx.view(-1)))

    return (t1 + t2 + t3 + t4).mean()
    
def rejector_loss(y_hat, m, r_hat, target):
    """
    This loss function is specifically designed for the second stage of
    the learning-to-defer pipeline, as it relies on pre-computed model
    predictions rather than raw logits.

    Suitable only for use in the second training stage since it depends
    on pre-computed model predictions.

    Parameters
    ----------
    y_hat : torch.Tensor
        Model predictions (class indices), shape (B,) or (B, 1)
    m : torch.Tensor
        Human expert predictions (class indices), shape (B,) or (B, 1)
    r_hat : torch.Tensor
        Deferral policy model logits, shape (B, 2) where:
        - r_hat[:, 0]: logit for model prediction
        - r_hat[:, 1]: logit for human expert prediction
    target : torch.Tensor
        Ground truth labels (class indices), shape (B,) or (B, 1)

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the computed loss value.    
    
    """
    batch_size = y_hat.shape[0]

    assert y_hat.shape == (batch_size, ) or y_hat.shape == (batch_size, 1)
    assert m.shape == (batch_size, ) or m.shape == (batch_size, 1)
    assert r_hat.shape == (batch_size, 2)
    assert target.shape == (batch_size, ) or target.shape == (batch_size, 1)
    
    weights = F.softmax(r_hat, -1)
    tmp = torch.cat([(y_hat != target).reshape(-1, 1),
                     (m != target).reshape(-1, 1)], -1)
    return(tmp * weights).sum(-1).mean()
