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

def test_mean_index():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    assert mvn.mean_index(0) == 6
    assert mvn.mean_index(1) == 7
    assert mvn.mean_index(2) == 8

def test_var_index():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    assert mvn.var_index(0) == 0
    assert mvn.var_index(1) == 2
    assert mvn.var_index(2) == 5

def test_covar_index():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    assert mvn.var_index(0, 1) == 1
    assert mvn.var_index(0, 2) == 3
    assert mvn.var_index(1, 0) == 1
    assert mvn.var_index(1, 2) == 4
    assert mvn.var_index(2, 0) == 3
    assert mvn.var_index(2, 1) == 4

