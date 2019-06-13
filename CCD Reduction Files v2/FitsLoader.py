from Data import Data
from errors import FitsLoaderError

from astropy.io import fits as fits
import numpy as np


class FitsLoader(object):
    """FitLoader can open a fits file, unpack it and returns a
    Data object storing the data and the exposure time.
    """

    def __init__(self, filename):
        """Loads a Fits file, creates and returns a Data object."""
        try:
            assert isinstance(filename, str), "filepath must be a string"
        except AssertionError as excep:
            raise FitsLoaderError(excep) from excep

        self.data = Data([], [], [])
        self._opener(filename)

    def _opener(self, filename):
        """Opens the file and sends it to be unpacked."""
        try:
            with fits.open(filename) as _file:
                self._unpack(_file)
        except Exception as excep:
            raise FitsLoaderError(excep) from excep

    def _unpack(self, _file):
        """Unpacks the fits file and stores the relevant."""
        data = []
        time = []
        header = []
        for i in _file:
            try:
                header.append(i.header)
                if i.data is not None:
                    data.append(i.data.astype(np.float64))
                if "exptime" in i.header:
                    time.append(i.header["exptime"])

            except Exception as excep:
                raise FitsLoaderError(excep) from excep
        self.data = Data(data, time, [None], header)
