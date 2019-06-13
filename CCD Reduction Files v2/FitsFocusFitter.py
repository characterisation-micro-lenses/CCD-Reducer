# -*- coding: utf-8 -*-
from FitsFocus import FitsFocus
from CCDFocusFitter import CCDFocusFitter
from FitsFolderLaserReducer import FitsFolderLaserReducer
from errors import FitsFocusFitterError


class FitsFocusFitter(CCDFocusFitter):

    @staticmethod
    def _error(exception=None):
        """Raises the CCDFocusFitterError."""
        return FitsFocusFitterError(exception)

    @staticmethod
    def cube(folderpath, masterpath, savepath, pixel_size, delta, realign):
        f =  FitsFolderLaserReducer(folderpath, masterpath, savepath, pixel_size)
        return f.cube(delta, realign)

    @staticmethod
    def zposition(names):
        """This function changes names to position.
        This is done by discarding the first 6 characters.
        This function thus only works if the files are named: Focus 13.140.
        Something similar will of course also work.
        If you use a different name, change this function accordingly!
        """

        z = []
        for i in names:
            z.append(float(i[6:]) * 1e-3)  # assumed that the namee is Focus (z) with (z) f.e.: 13.530
        z = np.flip(np.min(z) - np.array(z))  # flips z because increasing z is moving to the laser!
        return z
