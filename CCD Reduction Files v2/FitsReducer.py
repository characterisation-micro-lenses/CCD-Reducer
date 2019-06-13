# -*- coding: utf-8 -*-
from CCDReducer import CCDReducer
from FitsLoader import FitsLoader
from errors import FitsReducerError

class FitsReducer(CCDReducer):
    """CCDReducer specifically for Fits files."""

    @staticmethod
    def _loadfile(filepath):
        f =  FitsLoader(filepath)
        return f.data

    @staticmethod
    def _error(exception=None):
        """Raises the FitsReducerError."""
        return FitsReducerError(exception)
