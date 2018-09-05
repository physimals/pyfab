import numpy as np

from fabber.mvn import MVN

def test_one_param():
    d = np.zeros((5, 5, 5, 3))
    mvn = MVN(d)
    assert mvn.nparams == 1

def test_two_params():
    d = np.zeros((5, 5, 5, 6))
    mvn = MVN(d)
    assert mvn.nparams == 2

def test_three_params():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    assert mvn.nparams == 3

