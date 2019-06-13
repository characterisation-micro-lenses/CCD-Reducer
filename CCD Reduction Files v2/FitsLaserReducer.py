from CCDLaserReducer import CCDLaserReducer
from FitsLoader import FitsLoader
from errors import FitsLaserReducerError


class FitsLaserReducer(CCDLaserReducer):
    """LaserReducer class specifically for Fits files."""

    def __init__(self, filepath, masterpath=None, savepath=None, pixel_size=9e-6, magnification=100):
        super(FitsLaserReducer, self).__init__(filepath, masterpath, savepath, pixel_size, magnification)

    @staticmethod
    def _error(exception=None):
        """Raises the FitsLaserReducerError."""
        return FitsLaserReducerError(exception)

    @staticmethod
    def _loadfile(filepath):
        f = FitsLoader(filepath)
        return f.data
