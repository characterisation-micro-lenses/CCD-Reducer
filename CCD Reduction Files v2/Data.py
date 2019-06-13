import numpy as np


class DataError(Exception):
    """Error class for the Data class"""
    pass


class Data(object):
    def __str__(self):
        string = ""
        for i in self:
            string += "("
            string += str(i.data()[0]) + ",\n"
            string += str(i.time()[0]) + ", "
            string += str(i.id()[0]) + "),\n\n"
        string += "rest:\n" + str(self.__rest)
        return string

    def __repr__(self):
        string = "Data("
        string += repr(self.__data) + ", "
        string += repr(self.__time) + ", "
        string += repr(self.__id) + ", "
        string += repr(self.__rest) + ")"
        return string

    def _equivalent(self, y):
            assert isinstance(y, Data), "You can only add or subtract two Data objects."
            assert len(self) == len(y), "The two Data objects must have the same sizes."
            assert self.shape == y.shape, "The two Data objects must have the same shapes."
            assert self.__id == y.id(), "The id must be equal."

    def _combineability(self, y):
        if isinstance(y, Data):
            self._equivalent(y)
            return y.data(), y.time()
        elif isinstance(y, (int, float)):
            return [y] * len(self), [y] * len(self)
        elif isinstance(y, (tuple, list, np.ndarray)):
                assert len(self) == len(y), "The lengths must match."
                return y, y
        else:
            raise AssertionError("y is not a float, int, tuple, list, ndarray or Data object")

    def __add__(self, y):
        try:
            self._equivalent(y)
        except AssertionError as e:
            raise DataError("Cannot add these two: " + str(e)) from e

        x_data, y_data = self.__data, y.data()
        x_time, y_time = self.__time, y.time()
        _data = []
        for i in range(len(self)):
            if y_time[i] == 0:
                _data.append(x_data[i] + y_data[i])
            else:
                _data.append(x_data[i] + y_data[i] * x_time[i] / y_time[i])

        return Data(_data, self.__time, self.__id)


    def __sub__(self, y):
        try:
            self._equivalent(y)
        except AssertionError as e:
            raise DataError("Cannot subtract these two: " + str(e)) from e

        x_data, y_data = self.__data, y.data()
        x_time, y_time = self.__time, y.time()
        _data = []
        for i in range(len(self)):
            if y_time[i] == 0:
                _data.append(x_data[i] - y_data[i])
            else:
                _data.append(x_data[i] - y_data[i] * x_time[i] / y_time[i])

        return Data(_data, self.__time, self.__id)

    def __mul__(self, y):
        try:
            y_data, y_time = self._combineability(y)
        except AssertionError as e:
            raise DataError("Cannot multiply these two: " + str(e)) from e

        _data, _time = [], []
        x_data, x_time = self.__data, self.__time
        for i in range(len(self)):
            _data.append(x_data[i] * y_data[i])
            _time.append(x_time[i] * y_time[i])

        return Data(_data, _time, self.__id)

    def __truediv__(self, y):
        try:
            y_data, y_time = self._combineability(y)
        except AssertionError as e:
            raise DataError("Cannot devide these two: " + str(e)) from e

        _data, _time = [], []
        x_data, x_time = self.__data, self.__time
        for i in range(len(self)):
            _data.append(x_data[i] / y_data[i])
            _time.append(x_time[i] / y_time[i])

        return Data(_data, _time, self.__id)

    def __pow__(self, y):
        """When you a Data object to the power of another Data object,
        you insert the second dataset after the first.
        A Data object to the power of an integer (n) returns a Data object
        containing n times the Data object.
        """
        try:
            assert isinstance(y, Data) or isinstance(y, int), "y must be a Data object or an integer."
            if isinstance(y, Data):
                y_data = y.data()
                y_time = y.time()
                y_id = y.id()
            else:
                assert y > 1, "y must be a positive integer of at least 2."
                y_data = self.__data * (y - 1)
                y_time = self.__time * (y - 1)
                y_id = self.__id * (y - 1)
        except AssertionError as e:
            raise DataError("Cannot combine these two: " + str(e)) from e

        return Data(self.__data + y_data, self.__time + y_time, self.__id + y_id)

    def __eq__(self, y):
        if not isinstance(y, Data):
            return False
        if self.shape == y.shape and len(self) == len(y):
            _bool = [self.rest() == y.rest()]
            for i in range(len(self)):
                _bool.append(np.all(self.__data[i] == y.data()[i]))
                _bool.append(np.all(self.__time[i] == y.time()[i]))
                _bool.append(self.__id[i] == y.id()[i])
            return bool(np.all(_bool))
        else:
            return False

    def __ne__(self, y):
        if self.__eq__(y):
            return False
        else:
            return True

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, key):
        _data = self.__data[key]
        _time = self.__time[key]
        _id = self.__id[key]
        if isinstance(_data, tuple):
            return Data(_data, _time, _id)
        elif isinstance(_data, np.ndarray):
            return Data([_data], [_time], [_id])
        else:
            raise DataError("Something went wrong!")

    def __init__(self, data, time, id, rest=None):
        try:
            assert len(data) == len(time) == len(id), "The lists lengths must match!"
            self._check_and_set_data(data)
            self._check_and_set_time(time)
            self._check_and_set_id(id)
            self.shape = self._shape()
            self.__rest = rest
        except Exception as e:
            raise DataError(e) from e

    def _check_and_set_data(self, data):
        self._checklistness(data, "data")
        for i in data:
            assert isinstance(i, np.ndarray), "The data list must contain numpy arrays"
        self.__data = tuple(data)

    def _check_and_set_time(self, time):
        self._checklistness(time, "time")
        for i in time:
            assert isinstance(i, (int, float)), "The time list must contain integers or floats"
        self.__time = tuple(time)

    def _check_and_set_id(self, id):
        self._checklistness(id, "id")
        self.__id = tuple(id)

    @staticmethod
    def _checklistness(obj, name):
        assert isinstance(obj, (tuple, list, np.ndarray)), "The {} must be a tuple, list or array".format(name)

    def _shape(self):
        """Returns a list containing the shape of all data-arrays"""
        totalshape = []
        for i in self.__data:
            totalshape.append(i.shape)
        return tuple(totalshape)

    def data(self):
        """Returns a tuple containging the data-arrays."""
        return self.__data

    def time(self):
        """Returns a tuple containing the times."""
        return self.__time

    def id(self):
        """Returns a tuple containing the id."""
        return self.__id

    def rest(self):
        """Returns rest"""
        return self.__rest

    def medians(self):
        """Returns a tuple containing the medians of the various
        data-arrays.
        """
        _median = []
        for i in self.__data:
            _median.append(np.median(i))
        return tuple(_median)

    def maximums(self):
        """Returns a tuple containing the maximum values of the various
        data-arrays.
        """
        _max = []
        for i in self.__data:
            _max.append(np.max(i))
        return tuple(_max)

    def minimums(self):
        """Returns a tuple containing the minimum values of the various
        data-arrays.
        """
        _min = []
        for i in self.__data:
            _min.append(np.min(i))
        return tuple(_min)

    def moment(self, m):
        """Returns a tuple containing the normalised central moments of order m
        of the data-arrays. If m=1 returns the mean and if m=2 returns the
        variance. For any other m if the variance is 0, returns np.nan.
        """
        _moment = []
        for i in self.__data:
            mu = np.mean(i)
            std = np.std(i)
            if m == 1:
                _moment.append(mu)
            elif m == 2:
                _moment.append(std**2)
            elif std != 0:
                _moment.append(np.mean(((i - mu) / std)**m))
            else:
                _moment.append(np.nan)
        return tuple(_moment)
