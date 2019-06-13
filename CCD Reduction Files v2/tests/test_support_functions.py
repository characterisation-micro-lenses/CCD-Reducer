import support_functions as sf
import numpy as np
import pytest
import os

def test_gaus():
    sigma = 2.456
    y = sf.gauss(sigma, 1, sigma, 0, 0)
    assert y == np.exp(-0.5)
    y = sf.gauss(2 * sigma, 1, sigma, 0, 0)
    assert y == np.exp(-2)
    A = 3.3
    x0 = 2.3
    y0 = 10.220005
    y = sf.gauss(x0, A, 1, x0, y0)
    assert y == A + y0

def test_airy():
    A = 14.342
    x0 = 3.23
    y0 = 10.220005
    y = sf.airy(x0, A, 1, x0, y0)
    assert y == A + y0

    j1_1 = 0.4400505857449 #  value of J1(1)
    w = 2.434
    y = sf.airy(w, 1, w, 0, 0)
    assert np.abs(y - 4 * j1_1**2) < 1e-8

    j1_2 = 0.57672480775687  # value of J1(2)
    y = sf.airy(2 * w, 1, w, 0, 0)
    assert np.abs(y - j1_2**2) < 1e-8

def test_moment():
    a = [1, 2, 3]
    assert sf.moment(a, 1) == 2
    assert sf.moment(a, 2) == 2 / 3
    for i in range(30):
        assert sf.moment(a, i, [2, 1, 1]) == sf.moment(a, i, [0.5, 0.25, 0.25])
    w = [2, 1, 1]
    assert sf.moment(a, 1, w) == 7 / 4
    assert sf.moment(a, 2, w) == 11 / 16

def test_intersect():
    a = [0, 0, 1, 1, 0, 0]
    b = [1, 1, 0, 0, 1, 1]
    idx = sf.intersect(a, b)
    assert np.all(idx == [1, 3])

    a = [0, 0, 0, 1, 1, 1]
    assert np.all(sf.intersect(a, 0.5) == [2])

    with pytest.raises(AssertionError):
        sf.intersect([1, 2, 3], [1, 2, 3, 4])

def test_extender():
    a = sf.extender([0, 1])
    assert np.all(a == np.linspace(0, 1, 101))
    a = sf.extender([0, 1, 2])
    assert np.all(np.abs(a - np.linspace(0, 2, 201)) < 1e-8)

def test_aperturized():
    r = 100
    a = np.ones((2 * r + 5, 2 * r + 5))
    maskop = sf.aperturized(a.shape, (r+2, r+2), r)
    b = a * maskop
    assert np.abs(1 - np.sum(b) / (np.pi * r**2)) < 1e-3

    maskop2 = sf.aperturized(a.shape, (r+2, r+2), r, sharpedge=True)
    c = a * maskop2
    assert np.abs(1 - np.sum(c) / np.sum(b)) < 1e-3

    assert maskop.shape == maskop2.shape == a.shape

def test_aperture():
    r = 50
    a = np.ones((2 * r + 5, 2 * r + 5))
    maskop = sf.aperturized(a.shape, (r+2, r+2), r, sharpedge=True)
    mask = sf.aperture(a.shape, (r+2, r+2), r)
    assert np.sum(np.abs(maskop - mask)) == 0

def test_shapetester():
    a = [np.zeros((2, 2)), np.zeros((2, 2))]
    sf.shapetester(a)
    a = tuple(a)
    sf.shapetester(a)
    a = np.array(a)
    sf.shapetester(a)

    a = [np.zeros((2, 2)), np.zeros((3, 2))]
    with pytest.raises(AssertionError):
        sf.shapetester(a)

    sf.shapetester([np.zeros((2, 2))])
    with pytest.raises(AssertionError):
        sf.shapetester([])

def test_does_file_exist():
    path = os.path.dirname(__file__)
    assert sf.does_file_exist(path, "test_support_functions.py")
    assert not sf.does_file_exist(path, "doesntexist.dumb")

def test_find_free_filename():
    path = os.path.dirname(__file__)
    assert sf.find_free_filename(path, "test_support_functions", ".py") == path + "test_support_functions(1).py"
    assert sf.find_free_filename(path, "doesntexist", ".dumb") == path + "doesntexist.dumb"
