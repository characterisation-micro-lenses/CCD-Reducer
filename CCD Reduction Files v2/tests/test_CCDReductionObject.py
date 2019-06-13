from CCDReductionObject import CCDReductionObject, CCDBias, CCDDark, CCDFlat, CCDReductionObjectError
from Data import Data
import numpy as np
import pytest


def test_init_RO():
    f = CCDReductionObject("notarealpath")
    assert f.masterpath == f.filespath
    f = CCDReductionObject("notarealpath", "notarealpath2")
    with pytest.raises(CCDReductionObjectError):
        CCDReductionObject(1323)


def test_load_RO():
    f = CCDReductionObject("notarealpath")
    dat = f.load("notarealfile")
    assert dat is None or isisnstance(dat, Data)

def test_loadfile_RO():
    f = CCDReductionObject("notarealpath")
    assert isinstance(f._loadfile("notarealfile"), Data)

def test_check_lengths_RO():
    f = CCDReductionObject("notarealpath")
    a = [(1, 2, 3), (1, 2, 3), (1, 2, 3)]
    f._check_lengths(a)
    a = [(1, 2, 3), (1, 2, 3), (1, 2)]
    with pytest.raises(AssertionError):
        f._check_lengths(a)

def test_createbias():
    bias = []
    dd = Data([np.random.random((100, 100))], [1], [None])
    CCDBias._createbias(dd, bias)
    assert np.all(np.equal(bias, [dd.data()]))

def test_createdark():
    time = np.random.random()
    dd = Data([np.random.random((100, 100))], [time], [None])
    result = [(dd / time).data()]
    dark = []
    CCDDark._createdark(dd, None, dark)
    assert np.all(np.equal(dark, result))
    bias = Data([np.random.random((100, 100))], [0], [None])
    result = [((dd - bias) / time).data()]
    dark = []
    CCDDark._createdark(dd, bias, dark)
    assert np.all(np.equal(dark, result))

def test_createflat():
    pass
