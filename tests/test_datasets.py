import numpy as np

from haic.datasets import SyndataOne


def test_syndataone():
    data = SyndataOne(1000)
    assert len(data) == 1000

def test_syndataone_random():
    data1 = SyndataOne(100, random_state=2)
    data2 = SyndataOne(100, random_state=2)
    assert np.allclose(data1[0][0], data2[0][0])

def test_syndataone_human():
    data = SyndataOne(10)
    assert len(data[0]) == 3
    data = SyndataOne(10, include_human_labels=True)
    assert len(data[0]) == 3
    data = SyndataOne(10, include_human_labels=False)
    assert len(data[0]) == 2
