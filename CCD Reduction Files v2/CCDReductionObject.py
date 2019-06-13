from Data import Data
from errors import CCDReductionObjectError

import numpy as np
import pickle


class CCDReductionObject(object):
    """Abstract class for the Bias, Dark and Flat classes."""

    def __init__(self, masterpath, filespath=None):
        """Initiates the class. Loads masterpath (where the master file will
        be saved and loaded from) and an optional filespath (where the
        individual files will be loaded from). If filespath is not given, the
        files and masterpath are treated te be equal. This means that if
        filespath isn't given, the master file will be saved in the same
        path as the files.
        """

        try:
            assert isinstance(masterpath, str), "masterpath must be a string"
            self.masterpath = masterpath
            if filespath is None:
                self.filespath = masterpath
            else:
                assert isinstance(filespath, str), "filepath must be a string"
                self.filespath = filespath
        except AssertionError as excep:
            raise self._error(excep)

    @staticmethod
    def _loadfile(file):
        """Returns a Data object."""
        return Data([np.random.randint(5e3, 5e4, (1000, 1000))], [1], [None])

    @staticmethod
    def _error(exception=None):
        """Raises the CCDReducetionObjectError."""
        return CCDReductionObjectError(exception)

    @staticmethod
    def _check_lengths(list_of_lists):
        lengths = []
        for i in list_of_lists:
            lengths.append(len(i))
        assert np.all(np.array(lengths) == lengths[0]), "not all files have the same build."

    def create(self):
        """This function will be overwritten!"""
        pass

    def load(self, filename):
        """Loads the master file."""
        try:
            with open(self.masterpath + filename + ".pcl", "rb") as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            return None

    def _save_object(self, obj, filename):
        """Saves the master file"""
        with open(self.masterpath + filename + ".pcl", "wb") as f:
            pickle.dump(obj, f)

    def _openallfiles(self, function):
        """Loops through all files and execute a function.
        Overwrite this function!
        """
        data = self._loadfile(self.filespath)
        function(data)


class CCDBias(CCDReductionObject):
    """The Bias class can create and load master_bias files."""

    def create(self):
        """Creates the master_bias file from a bias dataset."""
        masterbias = []
        bias = []
        self._openallfiles(lambda data: self._createbias(data, bias))
        self._check_lengths(bias)
        for i in range(len(bias[0])):
            _bias = []
            for j in range(len(bias)):
                _bias.append(bias[j][i])
            masterbias.append(np.median(_bias, axis=0))

        bias = Data(masterbias, [0] * len(masterbias), [None] * len(masterbias))
        self._save_object(bias, "master_bias")

    @staticmethod
    def _loadfile(file):
        """Returns a Data object."""
        return Data([np.random.randint(50e2, 51e2, (1000, 1000))], [0], [None])

    @staticmethod
    def _createbias(data, bias):
        """The function that is executed on each individual bias file data."""
        bias.append(data.data())

    def load(self):
        """Loads the master_bias file."""
        return super(CCDBias, self).load("master_bias")


class CCDDark(CCDReductionObject):
    """The Dark class can create and load master_dark files."""

    def create(self):
        """Creates the master_dark file from a dark dataset."""
        masterdark = []
        dark = []
        bias = CCDBias(self.masterpath).load()
        self._openallfiles(lambda data: self._createdark(data, bias, dark))
        self._check_lengths(dark)
        for i in range(len(dark[0])):
            _dark = []
            for j in range(len(dark)):
                _dark.append(dark[j][i])
            masterdark.append(np.median(_dark, axis=0))

        dark = Data(masterdark, [1] * len(masterdark), [None] * len(masterdark))
        self._save_object(dark, "master_dark")

    @staticmethod
    def _loadfile(file):
        """Returns a Data object."""
        return Data([np.random.randint(3e4, 4e4, (1000, 1000))], [100], [None])

    @staticmethod
    def _createdark(data, bias, dark):
        """The function that is executed on each individual bias file data."""

        if bias is None:
            bias = Data([np.zeros(i) for i in data.shape], [0] * len(data), [None] * len(data))

        _dark = (data - bias) / data.time()
        dark.append(_dark.data())

    def load(self):
        """Loads the master_dark file."""
        return super(CCDDark, self).load("master_dark")


class CCDFlat(CCDReductionObject):
    """The Flat class can create and load master_flat files."""

    def create(self):
        """Creates the master_flat file from a flat dataset."""

        masterflat = []
        flat = []
        bias = CCDBias(self.masterpath).load()
        dark = CCDDark(self.masterpath).load()
        self._openallfiles(lambda data: self._createflat(data, bias, dark, flat))
        self._check_lengths(flat)
        for i in range(len(flat[0])):
            _flat = []
            for j in range(len(dark)):
                _flat.append(flat[j][i])
            _flat = np.median(_flat, axis=0)
            masterflat.append(_flat / np.mean(_flat))

        flat = Data(masterflat, [1] * len(masterflat), [None] * len(masterflat))
        self._save_object(flat, "master_flat")

    @staticmethod
    def _loadfile(file):
        """Returns a Data object."""
        return Data([np.random.randint(3e4, 4e4, (1000, 1000))], [1], [None])

    @staticmethod
    def _createflat(data, bias, dark, flat):
        """The function that is executed on each individual bias file data."""

        if bias is None:
            bias = Data([np.zeros(i) for i in data.shape], [0] * len(data), [None] * len(data))
        if dark is None:
            dark = Data([np.zeros(i) for i in data.shape], [1] * len(data), [None] * len(data))

        raw = (data - bias) - dark
        raw /= raw.moment(1)
        flat.append(raw.data())

    def load(self):
        """Loads the master_flat file."""
        return super(CCDFlat, self).load("master_flat")
