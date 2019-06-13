# -*- coding: utf-8 -*-
from CCDReductionObject import CCDReductionObject, CCDBias, CCDDark, CCDFlat
from FitsLoader import FitsLoader
from errors import FitsReductionObjectError
import os


class FitsReductionObject(CCDReductionObject):
    @staticmethod
    def _loadfile(file):
        """Returns a Data object."""
        f =  FitsLoader(file)
        return f.data

    @staticmethod
    def _error(exception=None):
        """Raises the CCDReducetionObjectError."""
        return FitsReductionObjectError(exception)

    def _openallfiles(self, function):
        """Loops through all files and execute a function."""
        for i in sorted(os.listdir(self.filespath)):
            if ".fit" in i:
                data = self._loadfile(self.filespath + i)
                function(data)


class FitsBias(FitsReductionObject, CCDBias):
    """The Bias class can create and load master_bias files."""
    pass


class FitsDark(FitsReductionObject, CCDDark):
    """The Dark class can create and load master_dark files."""
    pass


class FitsFlat(FitsReductionObject, CCDFlat):
    """The Flat class can create and load master_flat files."""
    pass
