# -*- coding: utf-8 -*-
from CCDFolderLaserReducer import CCDFolderLaserReducer
from FitsLaserReducer import FitsLaserReducer
from errors import FitsFolderLaserReducerError

class FitsFolderLaserReducer(CCDFolderLaserReducer):

    @staticmethod
    def _error(exception=None):
        """Raises the CCDFolderReducerError."""
        return FitsFolderLaserReducerError(exception)

    @staticmethod
    def _laserreducer(folderpath, masterpath, savepath, pixel_size):
        f = FitsLaserReducer(folderpath, masterpath, savepath, pixel_size)
        return f
