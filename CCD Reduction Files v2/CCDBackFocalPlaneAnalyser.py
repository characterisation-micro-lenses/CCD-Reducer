# -*- coding: utf-8 -*-
import support_functions as sf
from CCDReducer import CCDReducer
from errors import CCDBackFocalPlaneAnalyserError


from matplotlib import pyplot as plt, colors
import numpy as np


class CCDBackFocalPlaneAnalyser(object):
    """Master class to analyse two back focal plane images.

    __init__(self, lensfilepath, nolensfilename [, masterpath]):
        lensfilepath:
                The full path to the data file containing the Back Focal Plane
                image with the lens.
        nolensfilepath:
                The full path to the data file containing the BFP image without
                the lens.
        masterpath:
                The full path to the master files folder created by
                CCDReductionObject. If masterpath is not given the images are
                not reduced and the data has a larger error.


    Attributes:
        self.lens:
                Data object holding the lens BFP data
        self.nolens:
                Data object holding the nolens BFP data

    Functions:
        self.show(self [, x, y, [r]]):
                Displays the two images. Plots a horizontal and vertical
                line at y and x resp. If r is given also plots a circle
                centered at x, y
        self.findT(self, x, y, r [, plot=True]):
                Calculates the Transmissivity by integrating over a circular
                aperture centered on x, y with radius r. T is de fraction of
                lens to nolens. If plot is True also calls show to visualise
                where was integrated.
    """

    def __init__(self, lensfilepath, nolensfilepath, masterpath=None):
        self.lens = self._loadfile(lensfilepath, masterpath, None)
        self.nolens = self._loadfile(nolensfilepath, masterpath, None)
        self.lens /= self.lens.time()
        self.nolens /= self.nolens.time()
        assert len(self.lens) == len(self.nolens), "lens and nolens don't have an equal number of images."

    @staticmethod
    def _loadfile(filepath, masterpath, savepath):
        """Returns a Data object."""

        f = CCDReducer(filepath, masterpath, savepath)
        return f.data

    def show(self, x=None, y=None, r=None):
        """Displays the two images. Plots a horizontal and vertical line at y
        and x resp. If r is given also plots a circle centered at x, y
        """

        assert self._assertlengths(self.lens, x, y, r), "all lengths must match!"
        if x is None:
            x = [None] * len(self.lens)
        if y is None:
            y = [None] * len(self.lens)
        if r is None:
            r = [None] * len(self.lens)

        for i in range(len(self.lens)):
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            self._show(fig, ax1, self.lens.data()[i], x[i], y[i], r[i])
            self._show(fig, ax2, self.nolens.data()[i], x[i], y[i], r[i])
            ax2.set_xlabel("x position")
            ax1.set_ylabel("y position")
            ax2.set_ylabel("y position")

    @staticmethod
    def _show(fig, ax, data, x, y, r):
        im = ax.imshow(data, cmap="jet", interpolation="none", aspect="auto")
        if x is not None:
            ax.axvline(x, color="red")
        if y is not None:
            ax.axhline(y, color="blue")
        if r is not None and x is not None and y is not None:
            theta = np.linspace(0, 2 * np.pi, 1000)
            ax.plot(x + r * np.cos(theta), y + r * np.sin(theta), color="orange")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("counts/s")

    def findT(self, x, y, r, plot=True):
        """Calculates the Transmissivity by integrating over a circular
        aperture centered on x, y with radius r. T is de fraction of lens to
        nolens. If plot is True also calls show to visualise where was
        integrated.
        """

        assert len(x) == len(y) == len(r) == len(self.lens), "all lengths must match!"
        x, y, r = np.array(x), np.array(y), np.array(r)
        T = []
        for i in range(len(self.lens)):

            A = np.sum(self.lens.data()[i] * sf.aperturized(self.lens.data()[i].shape, [y[i], x[i]], r))
            B = np.sum(self.nolens.data()[i] * sf.aperturized(self.nolens.data()[i].shape, [y[i], x[i]], r))
            T.append(A / B)
        if plot is True:
            self.show(x, y, r)
        return T

    @staticmethod
    def _assertlengths(data, x, y, r):
        if x == y == r is None:
            return True
        _thing = []
        for i in [x, y, r]:
            if i is None:
                _thing.append(True)
            else:
                _thing.append(len(data) == len(i))
        return np.all(_thing)
