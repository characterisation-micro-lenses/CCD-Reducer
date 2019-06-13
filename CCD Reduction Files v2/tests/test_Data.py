from Data import Data, DataError
import numpy as np
import pytest

data1 = [np.array([[1, 1], [1, 4]])]
data1test = (np.array([[1, 1], [1, 4]]), )
time1 = [1]
time1test = (1, )
iden1 = ["blabla"]
iden1test = ("blabla", )

data2 = [np.array([[1, 1], [1, 4]]), np.array([2, 3])]
data2test = (np.array([[1, 1], [1, 4]]), np.array([2, 3]))
time2 = [1, 5.3]
time2test = (1, 5.3)
iden2 = ["blabla", "bloblo"]
iden2test = ("blabla", "bloblo")

rest = "This is a nice text."
resttest = "This is a nice text."


def test_creation():
    x = Data(data1, time1, iden1)
    assert x.shape == ((2, 2), )

    x = Data(data2, time2, iden2, rest)
    assert x.shape == ((2, 2), (2, ))

    with pytest.raises(DataError):
        Data("blabla", 1, 4)

    with pytest.raises(DataError):
        Data(data2, time2, ["Bla"])


def test_data():
    x = Data(data1, time1, iden1)
    assert np.all(np.equal(x.data(), data1test))
    assert isinstance(x.data(), tuple)

    x = Data(data2, time2, iden2)
    assert np.all(np.equal(x.data()[0], data2test[0]))
    assert np.all(np.equal(x.data()[1], data2test[1]))
    assert isinstance(x.data(), tuple)


def test_time():
    x = Data(data1, time1, iden1)
    assert x.time() == time1test
    assert isinstance(x.time(), tuple)

    x = Data(data2, time2, iden2)
    assert x.time() == time2test
    assert isinstance(x.time(), tuple)


def test_id():
    x = Data(data1, time1, iden1)
    assert x.id() == iden1test
    assert isinstance(x.id(), tuple)

    x = Data(data2, time2, iden2)
    assert x.id() == iden2test
    assert isinstance(x.id(), tuple)


def test_rest():
    x = Data(data1, time1, iden1, rest)
    assert x.rest() == resttest
    x = Data(data2, time2, iden2)
    assert x.rest() == None


def test_str():
    x = Data(data1, time1, iden1, rest)
    string = "("
    string += str(x.data()[0]) + ",\n"
    string += str(x.time()[0]) + ", "
    string += str(x.id()[0]) + "),\n\n"
    string += "rest:\n" + str(x.rest())
    assert str(x) == string
    x = Data(data2, time2, iden2, rest)
    string = "("
    string += str(x.data()[0]) + ",\n"
    string += str(x.time()[0]) + ", "
    string += str(x.id()[0]) + "),\n\n"
    string += "("
    string += str(x.data()[1]) + ",\n"
    string += str(x.time()[1]) + ", "
    string += str(x.id()[1]) + "),\n\n"
    string += "rest:\n" + str(x.rest())
    assert str(x) == string


def test_repr():
    x = Data(data2, time2, iden2, rest)
    string = "Data("
    string += repr(x.data()) + ", "
    string += repr(x.time()) + ", "
    string += repr(x.id()) + ", "
    string += repr(x.rest()) + ")"
    assert repr(x) == string


def test_len():
    x = Data(data1, time1, iden1)
    assert len(x) == 1

    x = Data(data2, time2, iden2)
    assert len(x) == 2

def test_getitem():
    x = Data(data1, time1, iden1)
    assert x[0] == x

    y = Data(data2, time2, iden2)
    assert len(y[0]) == 1
    assert y[0] == x


def test_medians():
    x = Data(data1, time1, iden1)
    assert x.medians() == (1, )
    x = Data(data2, time2, iden2)
    assert x.medians() == (1, 2.5)


def test_maximums():
    x = Data(data1, time1, iden1)
    assert x.maximums() == (4, )
    x = Data(data2, time2, iden2)
    assert x.maximums() == (4, 3)


def test_minimums():
    x = Data(data1, time1, iden1)
    assert x.minimums() == (1, )
    x = Data(data2, time2, iden2)
    assert x.minimums() == (1, 2)


def test_moment():
    x = Data(data1, time1, iden1)
    assert x.moment(0) == (1, )
    assert x.moment(1) == (7 / 4, )
    var1 = 3/4 * (1 - 7/4)**2 + (4 - 7/4)**2 / 4
    assert x.moment(2) == (var1, )

    x = Data(data2, time2, iden2)
    assert x.moment(0) == (1, 1)
    assert x.moment(1) == (7 / 4, 2.5)
    assert x.moment(2) == (var1, 0.25)


def test_eq():
    x = Data(data1, time1, iden1)
    assert (x == 0) is False
    y = Data(data1, time1, iden1)
    assert (x == y) is True
    y = Data(data1, time1, iden1, rest)
    assert (x == y) is False
    y = Data(data2, time2, iden2)
    assert (x == y) is False
    y = Data(data1, time1, ["bla"])
    assert (x == y) is False
    assert (x == x) is True
    x = Data(data2, time2, iden2)
    y = Data(data2, time2, iden2)
    assert (x == y) is True

def test_neq():
    x = Data(data1, time1, iden1)
    assert (x != 0) is True
    y = Data(data1, time1, iden1)
    assert (x != y) is False
    y = Data(data1, time1, iden1, rest)
    assert (x != y) is True
    y = Data(data2, time2, iden2)
    assert (x != y) is True
    y = Data(data1, time1, ["bla"])
    assert (x != y) is True
    assert (x != x) is False
    x = Data(data2, time2, iden2)
    y = Data(data2, time2, iden2)
    assert (x != y) is False

def test_add():
    x = Data(data1, time1, iden1)
    y = Data(data1, time1, iden1)
    assert x + y == Data([data1[0] + data1[0]], time1, iden1)

    y = Data([data1[0] * 2], [time1[0] * 2], iden1)
    assert x + y == Data([data1[0] + data1[0]], time1, iden1)

    y = Data([data1[0] * 2], [0], iden1)
    assert x + y == Data([data1[0] + 2 * data1[0]], time1, iden1)

    x = Data(data2, time2, iden2)
    y = Data(data2, time2, iden2)
    assert x + y == Data([data2[0] + data2[0], data2[1] + data2[1]], time2, iden2)

    y = Data([data2[0] * 2, data2[1] * 4], [time2[0] * 2, time2[1] * 2], iden2)
    assert x + y == Data([data2[0] + data2[0], data2[1] + 2 * data2[1]], time2, iden2)
    y = Data([data2[0] * 2, data2[1] * 4], [0, time2[1] * 2], iden2)
    assert x + y == Data([data2[0] + 2 * data2[0], data2[1] + 2 * data2[1]], time2, iden2)

    y = Data(data1, time1, iden1)
    with pytest.raises(DataError):
        x + y

    x = Data(data1, time1, iden1)
    y = Data([np.array([[0, 0], [0, 1]])], [0], iden1)
    assert x + y == Data([np.array([[1, 1], [1, 5]])], time1, iden1)

def test_sub():
    x = Data(data1, time1, iden1)
    y = Data(data1, time1, iden1)
    assert x - y == Data([np.array([[0, 0], [0, 0]])], time1, iden1)

    y = Data([data1[0] * 2], [time1[0] * 2], iden1)
    assert x - y == Data([np.array([[0, 0], [0, 0]])], time1, iden1)

    y = Data([data1[0] * 2], [0], iden1)
    assert x - y == Data([-data1[0]], time1, iden1)

    x = Data(data2, time2, iden2)
    y = Data(data2, time2, iden2)
    assert x - y == Data([np.array([[0, 0], [0, 0]]), np.array([0, 0])], time2, iden2)

    y = Data([data2[0] * 2, data2[1] * 4], [time2[0] * 2, time2[1] * 2], iden2)
    assert x - y == Data([np.array([[0, 0], [0, 0]]), -data2[1]], time2, iden2)
    y = Data([data2[0] * 2, data2[1] * 4], [0, time2[1] * 2], iden2)
    assert x - y == Data([-data2[0], -data2[1]], time2, iden2)

    y = Data(data1, time1, iden1)
    with pytest.raises(DataError):
        x - y

    x = Data(data1, time1, iden1)
    y = Data([np.array([[0, 0], [0, 1]])], [0], iden1)
    assert x - y == Data([np.array([[1, 1], [1, 3]])], time1, iden1)


def test_mul():
    x = Data(data1, time1, iden1)
    y = Data(data1, time1, iden1)
    assert x * y == Data([data1[0] * data1[0]], [time1[0] * time1[0]], iden1)
    assert x * 2 == Data([data1[0] * 2], [time1[0] * 2], iden1)

    x = Data(data2, time2, iden2)
    y = Data(data2, time2, iden2)
    assert x * y == Data([data2[0] * data2[0], data2[1] * data2[1]],
                         [time2[0] * time2[0], time2[1] * time2[1]], iden2)
    assert x * 2.25 == Data([data2[0] * 2.25, data2[1] * 2.25],
                         [time2[0] * 2.25, time2[1] * 2.25], iden2)

    y = Data(data1, time1, iden1)
    with pytest.raises(DataError):
        x * y


def test_div():
    x = Data(data1, time1, iden1)
    y = Data(data1, time1, iden1)
    assert x / y == Data([data1[0] / data1[0]], [time1[0] / time1[0]], iden1)
    assert x / 2 == Data([data1[0] / 2], [time1[0] / 2], iden1)

    x = Data(data2, time2, iden2)
    y = Data(data2, time2, iden2)
    assert x / y == Data([data2[0] / data2[0], data2[1] / data2[1]],
                         [time2[0] / time2[0], time2[1] / time2[1]], iden2)
    assert x / 2.25 == Data([data2[0] / 2.25, data2[1] / 2.25],
                         [time2[0] / 2.25, time2[1] / 2.25], iden2)

    y = Data(data1, time1, iden1)
    with pytest.raises(DataError):
        x / y


def test_pow():
    x = Data(data1, time1, iden1)
    y = Data(data1, time1, iden1)
    assert x ** y == Data(data1 + data1, time1 + time1, iden1 + iden1)
    assert x ** 2 == Data(data1 * 2, time1 * 2, iden1 * 2)

    x = Data(data2, time2, iden2)
    y = Data(data2, time2, iden2)
    assert x ** y == Data(data2 + data2, time2 + time2, iden2 + iden2)
    assert x ** 4 == Data(data2 * 4, time2 * 4, iden2 * 4)

    y = Data(data1, time1, iden1)
    assert x ** y == Data(data2 + data1, time2 + time1, iden2 + iden1)

    with pytest.raises(DataError):
        x ** 3.4
