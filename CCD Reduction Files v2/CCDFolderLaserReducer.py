from CCDLaserReducer import CCDLaserReducer
from errors import CCDFolderLaserReducerError

import numpy as np
import os


class CCDFolderLaserReducer(object):

    def __init__(self, folderpath, masterpath=None, savepath=None, pixel_size=9e-6):
        try:
            assert isinstance(folderpath, str), "folderpath must be a string"
            assert isinstance(pixel_size, (float, int)), \
                        "pixel_size must be an integer or a float"
            assert isinstance(masterpath, str) or masterpath is None, "masterpath must be a string"
            assert isinstance(savepath, str) or savepath is None, "masterpath must be a string"
            self.folderpath = folderpath
            self.pixel_size = pixel_size
            if masterpath is None:
                self.masterpath = os.path.dirname(folderpath) + "/"
            else:
                self.masterpath = masterpath
            if savepath is None:
                self.savepath = os.path.dirname(folderpath) + "/"
            else:
                self.savepath = savepath
        except AssertionError as excep:
            raise self._error(excep) from excep

        self.files = sorted(os.listdir(self.folderpath))

    @staticmethod
    def _error(exception=None):
        """Raises the CCDFolderReducerError."""
        return CCDFolderLaserReducerError(exception)

    @staticmethod
    def _laserreducer(folderpath, masterpath, savepath, pixel_size):
        f = CCDLaserReducer(folderpath, masterpath, savepath, pixel_size)
        return f

    def all_fit_saved(self, log=False):
        for i in self.files:
            if "fit" in i:
                name = i[:-4]
                if log is True:
                    name += "_log"
                f = self._laserreducer(self.folderpath + i, self.masterpath, self.savepath, self.pixel_size)
                f.imsave(savename=name, title=i[:-4], log=log)

    def all_fit_reduced(self, fit=True, log=False):
        for i in self.files:
            if "fit" in i:
                name = i[:-4] + "_Reduced"
                if log is True:
                    name += "_log"
                f =  self._laserreducer(self.folderpath + i, self.masterpath, self.savepath, self.pixel_size)
                f.slicesave(savename=name, title=i[:-4], fit=fit, log=log)

    def all_fit_power(self):
        for i in self.files:
            if "fit" in i:
                f = self._laserreducer(self.folderpath + i, self.masterpath, self.savepath, self.pixel_size)
                f.powersave(savename=i[:-4] + "_Power", title=i[:-4])

    def cube(self, delta, realign=False):
        cubes = []
        names = []
        for i in self.files:
            if "fit" in i:
                datacube = []
                names.append(i[:-4])
                f = self._laserreducer(self.folderpath + i, self.masterpath, self.savepath, pixel_size=self.pixel_size)
                if realign is False:
                    normed = (f.data / f.data.time()).data()
                    for i in range(len(normed)):
                        datacube.append(normed[i])
                else:
                    self._realign_every_file_and_append(datacube, f.data, delta)
                cubes.append(datacube)
        if realign is False:
            return self._align_total_cube(cubes, delta), names
        return np.array(cubes).transpose([1, 0, 2, 3]), names

    def _realign_every_file_and_append(self, datacube, dataobject, delta):
        normed = dataobject / dataobject.time()
        maxima = f.find_laser()
        for i in range(len(normed)):
            data = normed[i]
            self._check_bounds(maxima[i], delta, data.shape)
            xmin, xmax = maxima[i][1] - delta, maxima[i][1] + delta + 1
            ymin, ymax = maxima[i][0] - delta, maxima[i][0] + delta + 1
            datacube.append(data[ymin:ymax, xmin:xmax])

    def _align_total_cube(self, cubes, delta):
        cc = []
        for i in range(len(cubes[0])):
            maximum = np.argwhere(cubes == np.max(cubes))[0][2:]
            self._check_bounds(maximum, delta, cubes[i][0].shape)
            ccc = []
            for j in cubes:
                xmin, xmax = maximum[1] - delta, maximum[1] + delta + 1
                ymin, ymax = maximum[0] - delta, maximum[0] + delta + 1
                ccc.append(j[i][ymin:ymax, xmin:xmax])
            cc.append(ccc)
        return np.array(cc)

    def _check_bounds(self, maximum, delta, shape):
        try:
            assert maximum[0] > delta, "delta is to large"
            assert maximum[1] > delta, "delta is to large"
            assert shape[0] - maximum[0] > delta, "delta is to large"
            assert shape[1] - maximum[1] > delta, "delta is to large"
        except AssertionError as excep:
            raise self._error(excep) from excep
