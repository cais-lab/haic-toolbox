import torch

from haic.metrics import (
    optimal_coverage_accuracy_curve,
    ac_auc,
    nac_auc
)

import pytest


def test_optimal_coverage_accuracy_curve():
    y_p = torch.tensor([0, 0, 1, 1])
    m_p = torch.tensor([0, 1, 1, 0])
    y_t = torch.tensor([1, 1, 1, 1])
    co, ao = optimal_coverage_accuracy_curve(y_p, m_p, y_t)
    assert co.shape == (4, )
    assert ao.shape == (4, )
    for i in range(len(co) - 1):
        assert co[i] >= co[i + 1]
    # If cov = 1.0 (all samples to the model)
    assert co[0] == pytest.approx(1.0)
    assert ao[0] == pytest.approx(0.5)
    # If cov = 0.75
    # best assignment is:
    #   0 -> model (0)
    #   1 -> human (1)
    #   2 -> model (1)
    #   3 -> model (1)
    # hence, accuracy = 0.75
    assert co[1] == pytest.approx(0.75)
    assert ao[1] == pytest.approx(0.75)
    # If cov = 0.25
    # best assignment is:
    #   0 -> human (0)
    #   1 -> human (1)
    #   2 -> human (1)
    #   3 -> model (1)
    # hence, accuracy = 0.75
    assert co[2] == pytest.approx(0.25)
    assert ao[2] == pytest.approx(0.75)
    # and all-human
    assert co[3] == pytest.approx(0.0)
    assert ao[3] == pytest.approx(0.5)

def test_ac_auc():
    c = torch.tensor([1.0, 0.5, 0.0])
    a = torch.tensor([0.0, 1.0, 0.0])
    assert ac_auc(c, a) == pytest.approx(0.5)
    assert ac_auc(reversed(c), reversed(a)) == pytest.approx(0.5)
    
def test_nac_auc():
    # Corner case 1. nAC-AUC of optimal policy is 1
    y_p = torch.tensor([0, 0, 1, 1])
    m_p = torch.tensor([0, 1, 1, 0])
    y_t = torch.tensor([1, 1, 1, 1])
    co, ca = optimal_coverage_accuracy_curve(y_p, m_p, y_t)
    assert nac_auc(co, ca, y_p, m_p, y_t) == pytest.approx(1.0)
