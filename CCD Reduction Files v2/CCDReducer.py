from Data import Data
import support_functions as sf
from CCDReductionObject import CCDBias, CCDDark, CCDFlat
from errors import CCDReducerError


from matplotlib import pyplot as plt, colors as colors
import numpy as np
import os


class CCDReducer(object):
    """Loads a single file and reduces it using a Bias, Dark and Flat."""

    def __init__(self, filepath, masterpath=None, savepath=None):
        """Initiates the class, loads the file and creates a science."""
        self.data = self._loadfile(filepath)
        try:
            assert isinstance(masterpath, str) or masterpath is None, "masterpath must be a string"
            assert isinstance(savepath, str) or savepath is None, "savepath must be a string"
            if masterpath is None:
                self.masterpath = os.path.dirname(filepath) + "/"
            else:
                self.masterpath = masterpath
            if savepath is None:
                self.savepath = os.path.dirname(filepath) + "/"
            else:
                self.savepath = savepath
        except AssertionError as excep:
            raise self._error(excep) from excep

        self._science()

    @staticmethod
    def _loadfile(filename):
        """This function must return a Data object.
        Overwrite this!
        """
        return Data([np.random.randint(5e3, 5e4, (1000, 1000))], [1], [None])

    @staticmethod
    def _error(exception=None):
        """Raises the CCDReducerError.
        Overwrite this!
        """
        return CCDReducerError(exception)

    @staticmethod
    def _bias(filepath):
        """Loads the bias object. This should not need to be changed.
        But if you save the bias object differently: Overwrite this!"""
        f =  CCDBias(filepath)
        return f.load()

    @staticmethod
    def _dark(filepath):
        """Loads the dark object. This should not need to be changed.
        But if you save the dark object differently: Overwrite this!"""
        f =  CCDDark(filepath)
        return f.load()

    @staticmethod
    def _flat(filepath):
        """Loads the flat object. This should not need to be changed.
        But if you save the flat object differently: Overwrite this!"""
        f = CCDFlat(filepath)
        return f.load()

    def _science(self):
        """Creates a science using a datafile
        as well as the master bias, dark, flat.
        """

        data = self.data
        bias = self._bias(self.masterpath)
        dark = self._dark(self.masterpath)
        flat = self._flat(self.masterpath)
        if bias is None:
            bias = Data([np.zeros(i) for i in data.shape], [0] * len(data), [None] * len(data))
        if dark is None:
            dark = Data([np.zeros(i) for i in data.shape], [1] * len(data), [None] * len(data))
        else:
            dark_ = []
            for i in range(len(dark)):
                d2 = dark.data()[i]
                mask = d2 < 0
                d2[mask] = 0
                dark_.append(d2)
            dark = Data(dark_, dark.time(), dark.id())
        if flat is None:
            flat = Data([np.ones(i) for i in data.shape], [1] * len(data), [None] * len(data))

        a = (data - bias)
        b = a - dark
        self.data = b / flat

    def imshow(self, cmap="jet", log=False, title="Image of Data"):
        """Shows the calibrated data."""
        try:
            assert isinstance(title, str), "title must be a string"
        except AssertionError as excep:
            raise self._error(excep) from excep

        for i in range(len(self.data)):
            fig = plt.figure(figsize=[15, 10.5])
            ax = fig.add_subplot(111)
            ax.set_title("Exposure Time =" + str(round(self.data.time()[i], 6)))
            self._imshow(fig, ax, self.data.data()[i], cmap, log)
            if i == 0:
                fig.suptitle(title)
            else:
                fig.suptitle(title + " " + i)
            fig.show()

    def imsave(self, cmap="jet", log=False, title="Image of Data", savename="CCDPic", extension=".png"):
        """Saves the calibrated data."""
        try:
            assert isinstance(title, str), "title must be a string"
            assert isinstance(savename, str), "filename must be a string"
            assert isinstance(extension, str), "extension must be a string"
        except AssertionError as excep:
            raise self._error(excep) from excep

        for i in range(len(self.data)):
            fig = plt.figure(figsize=[15, 10.5])
            ax = fig.add_subplot(111)
            ax.set_title("Exposure Time =" + str(round(self.data.time()[i], 6)))
            self._imshow(fig, ax, self.data.data()[i], cmap, log)
            if i == 0:
                fig.suptitle(title)
            else:
                fig.suptitle(title + " " + i)
            fig.savefig(sf.find_free_filename(self.savepath, savename, extension))
            plt.close(fig)

    @staticmethod
    def _imshow(fig, ax, data, cmap, log):
        extent = [-0.5, (data.shape[1] - 0.5), (data.shape[0] - 0.5), -0.5]
        if log is False:
            im = ax.imshow(data, extent=extent, cmap=cmap, aspect="equal")
        else:
            im = ax.imshow(data, extent=extent, cmap=cmap, aspect="equal", norm=colors.LogNorm())
        ax.set_xlabel(r"x position")
        ax.set_ylabel(r"y position")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Counts")
