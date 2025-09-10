import numpy as np 
import torch

from haic.losses import (
    rejector_loss,
    softmax_parametrization_loss,
    l_ce_loss,
    ova_loss
)

import pytest


def test_rejector_loss():
    y_hat = torch.tensor([0, 1, 0, 1])
    m = torch.tensor([1, 1, 0, 0])
    r_hat = torch.tensor([[0.1, -0.1],
                          [-0.1, 0.1],
                          [0.1, 0.2],
                          [1, 2]])
    y = torch.tensor([0, 0, 0, 0])
    loss = rejector_loss(y_hat, m, r_hat, y)
 
# -------

class Criterion:
    """Авторская реализация OvA из https://github.com/rajevv/OvA-L2D
    
    Для совместимости, по сравнению с версией, описанной в статье, они
    ввели коэффициент alpha.
    """

    def __init__(self):
        pass
        
    def softmax(self, outputs, m, labels, m2, n_classes):
        '''
        The L_{CE} loss implementation for CIFAR
        ----
        outputs: network outputs
        m: cost of deferring to expert cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
        labels: target
        m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
        n_classes: number of classes
        '''
        batch_size = outputs.size()[0]  # batch_size
        rc = [n_classes] * batch_size
        outputs = -m * torch.log2(outputs[range(batch_size), rc]) - m2 * torch.log2(
            outputs[range(batch_size), labels])  
        return torch.sum(outputs) / batch_size

    def ova(self, outputs, m, labels, m2, n_classes):
        """
        Параметры m и m2 расчитываются следующим образом:
        m - это просто признак того, что человек-эксперт дал правильный ответ;
        m2 - стоимость (возможно, предсказания моделью), которая равна 
            alpha [0; 1], если эксперт дал правильный ответ, и 1, если нет.
            Идея, по видимому, в следующем - если эксперт дал правильный ответ,
            то оптимизировать члены функции потерь, отвечающие за ошибку классификатора,
            не так важно (поэтому вес для этих членов берется меньше 1), если же
            эксперт ошибся, то модели важно хорошо отработать на этом образце,
            поэтому модель вес берется 1.
    
        См. также: https://github.com/rajevv/OvA-L2D/blob/main/main.py
        """
        batch_size = outputs.size()[0]
        l1 = Criterion.LogisticLoss(outputs[range(batch_size), labels], 1)
        l2 = torch.sum(Criterion.LogisticLoss(outputs[:,:n_classes], -1), dim=1) - Criterion.LogisticLoss(outputs[range(batch_size),labels],-1)
        l3 = Criterion.LogisticLoss(outputs[range(batch_size), n_classes], -1)
        l4 = Criterion.LogisticLoss(outputs[range(batch_size), n_classes], 1)

        l5 = m * (l4 - l3)

        l = m2 * (l1 + l2) + l3 + l5

        return torch.mean(l)

    @staticmethod
    def LogisticLoss(outputs, y):
        outputs[torch.where(outputs==0.0)] = (-1*y)*(-1*np.inf)
        l = torch.log2(1 + torch.exp((-1*y)*outputs))
        return l

def test_ova_loss():
    # Просто случайный пример
    hx = (torch.rand((5, 6)) - 0.5) * 5
    m = torch.tensor([0, 1, 2, 3, 4])
    rx = (torch.rand((5, )) - 0.5) * 5
    target = torch.tensor([0, 1, 2, 4, 4])

    loss1 = ova_loss(hx, m, rx, target)
    # Вычисляем авторскую реализацию при alpha = 1.0
    alpha = 1.0
    loss2 = Criterion().ova(
        torch.cat([hx, rx.view(-1, 1)], -1),
        (m == target).to(torch.float32), 
        target,
        torch.where(m == target, alpha, 1),
        hx.shape[1]
    )
    assert loss1.item() == pytest.approx(loss2.item())

def test_l_ce_loss():
    # Просто случайный пример
    hx = (torch.rand((5, 6)) - 0.5) * 5
    m = torch.tensor([0, 1, 2, 3, 4])
    rx = (torch.rand((5, )) - 0.5) * 5
    target = torch.tensor([0, 1, 2, 4, 4])

    loss1 = l_ce_loss(hx, m, rx, target)
    # Вычисляем реализацию, использованную в тестах
    # Veerma, при alpha = 1.0
    alpha = 1.0
    loss2 = Criterion().softmax(
        torch.softmax(torch.cat([hx, rx.view(-1, 1)], -1), -1),
        (m == target).to(torch.float32), 
        target,
        torch.where(m == target, alpha, 1),
        hx.shape[1]
    )
    assert loss1.item() == pytest.approx(loss2.item())

def test_softmax_parametrization_loss():
    # Просто случайный пример
    hx = (torch.rand((5, 6)) - 0.5) * 5
    m = torch.tensor([0, 1, 2, 3, 4])
    rx = (torch.rand((5, )) - 0.5) * 5
    target = torch.tensor([0, 1, 2, 4, 4])

    loss1 = softmax_parametrization_loss(hx, m, rx, target)
    # Вычисляем реализацию, использованную в тестах
    # Veerma, при alpha = 1.0
    alpha = 1.0
    loss2 = Criterion().softmax(
        torch.softmax(torch.cat([hx, rx.view(-1, 1)], -1), -1),
        (m == target).to(torch.float32), 
        target,
        torch.where(m == target, alpha, 1),
        hx.shape[1]
    )
    assert loss1.item() == pytest.approx(loss2.item())
