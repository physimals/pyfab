import pytest
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

def test_mean_index_name():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d, param_names=["a", "b", "c"])
    assert mvn.mean_index("a") == 6
    assert mvn.mean_index("b") == 7
    assert mvn.mean_index("c") == 8

def test_var_index():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    assert mvn.var_index(0) == 0
    assert mvn.var_index(1) == 2
    assert mvn.var_index(2) == 5

def test_var_index_name():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d, param_names=["a", "b", "c"])
    assert mvn.var_index("a") == 0
    assert mvn.var_index("b") == 2
    assert mvn.var_index("c") == 5

def test_covar_index():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    assert mvn.var_index(0, 1) == 1
    assert mvn.var_index(0, 2) == 3
    assert mvn.var_index(1, 0) == 1
    assert mvn.var_index(1, 2) == 4
    assert mvn.var_index(2, 0) == 3
    assert mvn.var_index(2, 1) == 4

def test_covar_index_name():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d, param_names=["a", "b", "c"])
    assert mvn.var_index("a", "b") == 1
    assert mvn.var_index("a", "c") == 3
    assert mvn.var_index("b", "a") == 1
    assert mvn.var_index("b", "c") == 4
    assert mvn.var_index("c", "a") == 3
    assert mvn.var_index("c", "b") == 4

def test_param_names_correct():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d, param_names=["a", "b", "c"])
    assert mvn.param_names == ["a", "b", "c"]

def test_param_names_none():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    assert len(mvn.param_names) == 3

def test_param_names_too_few():
    d = np.zeros((5, 5, 5, 10))
    with pytest.raises(ValueError):
        assert MVN(d, param_names=["a", "b"])

def test_param_names_too_many():
    d = np.zeros((5, 5, 5, 10))
    with pytest.raises(ValueError):
        assert MVN(d, param_names=["a", "b", "c", "d"])

def test_update_mean_const():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    for param in range(3):
        mvn.update(param, mean=float(param+1))
    for param in range(3):
        assert np.all(mvn[..., mvn.mean_index(param)] == float(param+1))

def test_update_mean_arr():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    for param in range(3):
        img = np.random.rand(5, 5, 5)
        mvn.update(param, mean=img)
        assert np.all(mvn[..., mvn.mean_index(param)] == img)

def test_update_var_const():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    for param in range(3):
        mvn.update(param, var=float(param+1))
    for param in range(3):
        assert np.all(mvn[..., mvn.var_index(param)] == float(param+1))

def test_update_var_arr():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    for param in range(3):
        img = np.random.rand(5, 5, 5)
        mvn.update(param, var=img)
        assert np.all(mvn[..., mvn.var_index(param)] == img)

def test_update_mean_var_const():
    d = np.zeros((5, 5, 5, 10))
    mvn = MVN(d)
    for param in range(3):
        mvn.update(param, mean=float(param+10), var=float(param+1))
    for param in range(3):
        assert np.all(mvn[..., mvn.mean_index(param)] == float(param+10))
        assert np.all(mvn[..., mvn.var_index(param)] == float(param+1))
