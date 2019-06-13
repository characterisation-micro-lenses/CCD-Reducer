import support_functions as sf
from CCDLaserReducer import CCDLaserReducer
from CCDFolderLaserReducer import CCDFolderLaserReducer
from errors import CCDFocusError

from matplotlib import pyplot as plt, colors as colors
import numpy as np
import os



class CCDFocus(object):
    """Master class used to analyse a focus.

    __init__(self, folderpath [, masterpath, savepath, delta=150,
        pixel_size=1e-6, magnifiction=100, realign=False]):

        folderpath:
                The full path to the folder containing CCD images taken at
                various positions around the focus.
        masterpath:
                The full path to the master files folder created by
                CCDReductionObject. If masterpath is not given the images are
                not reduced and the data has a larger error.
        savepath:
                The full path to where the created images (by save functions)
                are to be saved.
        delta:
                The distance from the optical axis that is just taken into
                account. Distances large than delta will not be used.
                Distances shorter or equal will be used in the analysis.
                Delta must be shorter than the shortest distance to the edge
                every CCD image.
        pixel_size:
                The physical size of the pixels used by the CCD camera (in m).
        magnification:
                Any magnification that changes the image from the CCD camera to
                the image in the focus. pixel_size/magnification is equivalent
                to setting magnification to 1.
        realign:
                If realign is True, realign every CCD image before creating
                the focus. If it is False it does not. Realigning is buggy,
                but corrects for shifting laserlight (that is, the center of
                the image does not remain static).


    Attributes:
        delta:
                As defined above.
        pixel_size:
                The pixel_size in the focus (so pixel_size/magnification).
        savepath:
                As defined above.
        z:
                The spatial position of the individual images.
        focus:
                The 3D focus created by placing the CCD images behind each
                other. This is the object of interest.

    Functions:
        show(self [, title="Focus of the laser", log=False, both=False]):
                Shows a 2D slice of the focus along the direction of
                propagation.

    """

    def __init__(self, folderpath, masterpath=None, savepath=None, delta=150, pixel_size=9e-6,
                 magnification=100, realign=False):
        try:
            assert isinstance(folderpath, str), "folderpath must be a string"
            assert isinstance(pixel_size, (float, int)), \
                        "pixel_size must be an integer or a float"
            assert isinstance(masterpath, str) or masterpath is None, "masterpath must be a string"
            assert isinstance(savepath, str) or savepath is None, "masterpath must be a string"
            self.delta = delta
            self.pixel_size = pixel_size/magnification
            if savepath is None:
                self.savepath = os.path.dirname(folderpath) + "/"
            else:
                self.savepath = savepath
        except AssertionError as excep:
            raise self._error(excep) from excep
        focus, names = self._cube(folderpath, masterpath, savepath, pixel_size/magnification, delta, realign)
        self.z = self._zposition(names)
        self.focus = np.flip(focus, 1)

    @staticmethod
    def _cube(folderpath, masterpath, savepath, pixel_size, delta, realign):
        f =  CCDFolderLaserReducer(folderpath, masterpath, savepath, pixel_size)
        return f.cube(delta, realign)

    @staticmethod
    def _error(exception=None):
        """Raises the CCDFocusError.
        Overwrite this!"""
        return CCDFocusError(exception)

    @staticmethod
    def _zposition(names):
        """This function can change how the list of names is changed into a list of positions.
        Overwrite this!"""
        return np.array(names)

    def show(self, title="Focus of the laser", log=False, both=False):
        try:
            assert isinstance(title, str), "title must be a string"
        except AssertionError as excep:
            raise self.error(excep) from excep

        for i in self.focus:
            fig = plt.figure(figsize=[15, 10.5])
            if both is True:
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                ax1.set_title("Variation along x-axis")
                ax2.set_title("Variation along y-axis")
                sliced1 = i[:, self.delta, :].transpose()
                sliced2 = i[:, :, self.delta].transpose()
                self._show(sliced1, fig, ax1, log)
                self._show(sliced2, fig, ax2, log)
            else:
                ax = fig.add_subplot(111)
                sliced = i[:, self.delta, :].transpose()
                self._show(sliced, fig, ax, log)
            fig.suptitle(title)
            fig.show()

    def save(self, title="Image of Data", log=False, both=False, savename="FocusPic", extension=".png"):
        """Saves the calibrated data."""
        try:
            assert isinstance(title, str), "title must be a string"
            assert isinstance(savename, str), "filename must be a string"
            assert isinstance(extension, str), "extension must be a string"
        except AssertionError as excep:
            raise self.error(excep) from excep

        for i in self.focus:
            fig = plt.figure(figsize=[15, 10.5])
            if both is True:
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                ax1.set_title("Variation along x-axis")
                ax2.set_title("Variation along y-axis")
                sliced1 = i[:, self.delta, :].transpose()
                sliced2 = i[:, :, self.delta].transpose()
                self._show(sliced1, fig, ax1, log)
                self._show(sliced2, fig, ax2, log)
            else:
                ax = fig.add_subplot(111)
                sliced = i[:, self.delta, :].transpose()
                self._show(sliced, fig, ax, log)
            fig.suptitle(title)
            fig.savefig(sf.find_free_filename(self.savepath, savename, extension))
            plt.close(fig)


    def _show(self, data, fig, ax, log):
        ps = self.pixel_size * 1e6
        loc = np.argwhere(data == np.max(data))[0]
        x = self.z - self.z[loc[1]]
        dx = (x[-1] - x[0]) / ((len(x) - 1) * 2)
        xmin, xmax = (x[0] - dx) * 1e6, (x[-1] + dx) * 1e6
        ymax, ymin = (self.delta + 0.5) * ps, -(self.delta + 0.5) * ps
        extent = [xmin, xmax, ymax, ymin]
        if log is False:
            im = ax.imshow(data, cmap="jet", aspect="auto", extent=extent, interpolation="none")
        else:
            im = ax.imshow(np.abs(data), cmap="jet", aspect="auto", extent=extent, interpolation="none",
                           norm=colors.LogNorm())
        ax.scatter(x[loc[1]], (loc[0] - self.delta) * ps, marker="x", color="black")
        ax.set_xlim(*extent[:2])
        ax.set_ylim(*extent[2:])
        ax.set_xlabel(r"z ($\mu$m)")
        ax.set_ylabel(r"$\rho$ ($\mu$m)")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Counts/s")

    def imshow(self, zpos, title="CCD image", log=False):
        try:
            assert isinstance(zpos, int), "zpos must be an integer"
            assert isinstance(title, str), "title must be a string"
        except AssertionError as excep:
            raise self._error(excep)

        data = self.focus[:, zpos]
        z = self.z[zpos]
        for i in range(len(data)):
            axial_distr = self.focus[i, :, self.delta, self.delta]
            loc = np.argwhere(axial_distr == np.max(axial_distr)).reshape(-1)[0]
            fig = plt.figure(figsize=[15, 10.5])
            ax = fig.add_subplot(111)
            ax.set_title(r"z = {} $\mu$m".format(round((z - self.z[loc]) * 1e6, 2)))
            self._imshow(fig, ax, data[i], "jet", log)
            if i == 0:
                fig.suptitle(title)
            else:
                fig.suptitle(title + " " + i)
            fig.show()

    def imsave(self, zpos, log=False, title="Image of Data", savename="CCDPic", extension=".png"):
        try:
            assert isinstance(zpos, int), "zpos must be an integer"
            assert isinstance(title, str), "title must be a string"
            assert isinstance(savename, str), "filename must be a string"
            assert isinstance(extension, str), "extension must be a string"
        except AssertionError as excep:
            raise self._error(excep)

        data = self.focus[:, zpos]
        z = self.z[zpos]
        for i in range(len(data)):
            axial_distr = self.focus[i, :, self.delta, self.delta]
            loc = np.argwhere(axial_distr == np.max(axial_distr)).reshape(-1)[0]
            fig = plt.figure(figsize=[15, 10.5])
            ax = fig.add_subplot(111)
            ax.set_title(r"z = {} $\mu$m".format(round((z - self.z[loc]) * 1e6, 2)))
            self._imshow(fig, ax, data[i], "jet", log)
            if i == 0:
                fig.suptitle(title)
            else:
                fig.suptitle(title + " " + i)
            fig.savefig(sf.find_free_filename(self.savepath, savename, extension))
            plt.close(fig)

    def _imshow(self, fig, ax, data, cmap, log):
        ps = self.pixel_size * 1e6
        dt = self.delta + 0.5
        extent = [-dt * ps, dt * ps, dt * ps, -dt * ps]
        if log is False:
            im = ax.imshow(data, cmap=cmap, aspect="equal", extent=extent)
        else:
            im = ax.imshow(data, cmap=cmap, aspect="equal", extent=extent, norm=colors.LogNorm())
        ax.set_xlabel(r"x position ($\mu$m)")
        ax.set_ylabel(r"y position ($\mu$m)")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Counts/s")

    def sliceshow(self, zpos, title="Slice of Data", log=False, both=False, fit=False, overlap=False):
        try:
            assert isinstance(title, str), "title must be a string"
        except AssertionError as excep:
            raise self._error(excep) from excep

        data = self.focus[:, zpos]
        z = self.z[zpos]
        for i in range(len(data)):
            _data = data[i]
            axial_distr = self.focus[i, :, self.delta, self.delta]
            loc = np.argwhere(axial_distr == np.max(axial_distr)).reshape(-1)[0]
            fig = plt.figure(figsize=[15, 10.5])
            ax3 = fig.add_subplot(212)
            self._imshow(fig, ax3, _data, "jet", log)
            ax3.set_title(r"z = {} $\mu$m".format(round((z - self.z[loc]) * 1e6, 2)))
            ps = self.pixel_size * 1e6
            ax3.axhline(0, color="blue")
            if both is True:
                ax3.axvline(0, color="red")
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                self._slice1D(ax2, _data[:, self.delta], fit, ps, "red")
                ax2.set_ylim(bottom=_data[:, self.delta].min(), top=_data[:, self.delta].max())
                ax2.set_xlabel(r"position ($\mu$m)")
                if log is not False:
                    ax1.semilogy()
                    ax2.semilogy()
            else:
                ax1 = fig.add_subplot(211)
                if log is not False:
                    ax1.semilogy()
                if overlap is True:
                    ax3.axvline(0, color="red")
                    self._slice1D(ax1, _data[:, self.delta], fit, ps, "red")
            self._slice1D(ax1, _data[self.delta, :], fit, ps, "blue")
            if overlap is True:
                ymin = np.min([_data[self.delta, :].min(), _data[:, self.delta].min()])
                ymax = np.max([_data[self.delta, :].max(), _data[:, self.delta].max()])
            else:
                ymin = _data[self.delta, :].min()
                ymax = _data[self.delta, :].max()
            ax1.set_ylim(bottom=ymin, top=ymax)
            ax1.set_xlabel(r"position ($\mu$m)")
            ax1.set_ylabel("Counts/s")
            if i == 0:
                fig.suptitle(title)
            else:
                fig.suptitle(title + " " + str(i))

            fig.show()

    def slicesave(self, zpos, position=[None, None], title="Slice of Data", log=False, both=False, fit=False,
                  overlap=False, savename="Slice", extension=".png"):
        try:
            assert isinstance(title, str), "title must be a string"
            assert isinstance(savename, str), "savename must be a string"
            assert isinstance(extension, str), "extension must be a string"
        except AssertionError as excep:
            raise self._error(excep) from excep

        data = self.focus[:, zpos]
        z = self.z[zpos]
        for i in range(len(data)):
            _data = data[i]
            axial_distr = self.focus[i, :, self.delta, self.delta]
            loc = np.argwhere(axial_distr == np.max(axial_distr)).reshape(-1)[0]
            fig = plt.figure(figsize=[15, 10.5])
            ax3 = fig.add_subplot(212)
            self._imshow(fig, ax3, _data, "jet", log)
            ax3.set_title(r"z = {} $\mu$m".format(round((z - self.z[loc]) * 1e6, 2)))
            ps = self.pixel_size * 1e6
            ax3.axhline(0, color="blue")
            if both is True:
                ax3.axvline(0, color="red")
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                self._slice1D(ax2, _data[:, self.delta], fit, ps, "red")
                ax2.set_ylim(ymin=_data[:, self.delta].min(), ymax=_data[:, self.delta].max())
                ax2.set_xlabel(r"position ($\mu$m)")
                if log is not False:
                    ax1.semilogy()
                    ax2.semilogy()
            else:
                ax1 = fig.add_subplot(211)
                if log is not False:
                    ax1.semilogy()
                if overlap is True:
                    ax3.axvline(0, color="red")
                    self._slice1D(ax1, _data[:, self.delta], fit, ps, "red")
            self._slice1D(ax1, _data[self.delta, :], fit, ps, "blue")
            if overlap is True:
                ymin = np.min([_data[self.delta, :].min(), _data[:, self.delta].min()])
                ymax = np.max([_data[self.delta, :].max(), _data[:, self.delta].max()])
            else:
                ymin = _data[self.delta, :].min()
                ymax = _data[self.delta, :].max()
            ax1.set_ylim(ymin=ymin, ymax=ymax)
            ax1.set_xlabel(r"position ($\mu$m)")
            ax1.set_ylabel("Counts/s")
            if i == 0:
                fig.suptitle(title)
            else:
                fig.suptitle(title + " " + str(i))

            full_filename = sf.find_free_filename(self.savepath, savename, extension)
            fig.savefig(full_filename)
            plt.close(fig)

    def _slice1D(self, ax, sliced, fit, ps, color):
        x = np.arange(-self.delta, self.delta + 1) * ps
        ax.plot(x, sliced, color=color)
        if fit is True:
            self._add_fit_to_axis(ax, sliced, x, color)

    def _add_fit_to_axis(self, ax, fitdata, x, color):
        CCDLaserReducer._add_fit_to_axis(self, ax, fitdata, x, color)

    @staticmethod
    def _gaussian_fit(data, x):
        popt, success = CCDLaserReducer._gaussian_fit(data, x)
        return popt, success

    @staticmethod
    def _legend_fit_values(popt, success):
        labelstring = CCDLaserReducer._legend_fit_values(popt, success)
        return labelstring

    def characterise(self, plot=True):
        power, area, waist, z_R = [], [], [], []
        for i in self.focus:
            power_, area_, waist_, z_R_ = self._characterise_focus(i, plot)
            power.append(power_)
            area.append(area_)
            waist.append(waist_)
            z_R.append(z_R_)
        return self._focus_char_string(power, area, waist, z_R)

    def _characterise_focus(self, cube, plot):
        axial_profile = cube[:, self.delta, self.delta]
        z0 = int(np.argwhere(axial_profile == np.max(axial_profile)).reshape(-1)[0])
        x = np.arange(-self.delta, self.delta + 1) * self.pixel_size
        data = cube[z0, :, :]
        waistheight = np.max(cube) * np.exp(-2)  # intensity is E^2

        waistx, xlim = self._find_waist(data[self.delta, :], x, waistheight)
        waisty, ylim = self._find_waist(data[:, self.delta], x, waistheight)
        waistz, zlim = self._find_waist(axial_profile, self.z - self.z[z0], waistheight)
        power, area = self._focalpower(data, np.mean([waisty, waistx]))
        z_R = self._find_zR(cube, z0, x, plot)

        if plot is True:
            self._characterise_plotter(cube, z0, waistheight, xlim, ylim, zlim)

        return power, area, [waistz, waistx, waisty], z_R

    def _find_zR(self, cube, z0, x, plot):
        w = []
        for j in range(len(self.z)):
            d1, d2 = cube[j, self.delta, :], cube[j, :, self.delta]
            w1, _ = self._find_waist(d1, x, d1[self.delta] * np.exp(-2))
            w2, _ = self._find_waist(d2, x, d2[self.delta] * np.exp(-2))
            w.append(np.mean([w1, w2]))
        extend_w = sf.extender(w)
        idx = sf.intersect(extend_w, np.sqrt(2) * w[z0])
        extend_z = sf.extender(self.z - self.z[z0])
        if plot is True:
            fig = plt.figure(figsize=[15, 10.5])
            ax = fig.add_subplot(111)
            ax.plot(extend_z * 1e6, extend_w * 1e6)
            ax.axvline(extend_z[idx[0]] * 1e6, color="red", linestyle="dotted")
            ax.axvline(extend_z[idx[-1]] * 1e6, color="red", linestyle="dotted")
            ax.axhline(np.sqrt(2) * w[z0] * self.pixel_size, color="red", linestyle="dotted")
            ax.set_title(r"Axial distribution used for $z_R$")
            fig.show()
        return [np.abs(extend_z[idx[0]]), np.abs(extend_z[idx[-1]])]

    @staticmethod
    def _find_waist(sliced, x, height):
        extend_x, extended = sf.extender(x), sf.extender(sliced)
        idx = sf.intersect(extended, height)
        if len(idx) != 0:
            a, b = extend_x[idx[0]], extend_x[idx[-1]]
            return (b - a) / 2, [a, b]
        else:
            print("no waist found")
            return (x[-1] - x[0]) / 2, [x[0], x[-1]]

    def _focalpower(self, data, waist, subtract_median=True):
        if subtract_median is True:
            _data = data - np.median(data)  # subtract remaining background from pc screens and light switches.
        else:
            _data = data
        r = waist / self.pixel_size
        if r < self.delta:
            maskop = sf.aperturized(_data.shape, [self.delta, self.delta], r)
            power = np.sum(_data * maskop)
            area = self.pixel_size**2 * np.sum(maskop)
        else:
            power = -1
            area = -1
        return [power, area]

    def _characterise_plotter(self, cube, z0, height, xlim, ylim, zlim):
        x = np.arange(-self.delta, self.delta + 1) * self.pixel_size
        axial_profile = cube[:, self.delta, self.delta]
        data = cube[z0, :, :]
        fig = plt.figure(figsize=[15, 10.5])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(224)
        self._focus_plotter(ax1, axial_profile, self.z - self.z[z0], zlim[0], zlim[1], height)
        self._focus_plotter(ax2, data[self.delta, :], x, xlim[0], xlim[1], height)
        self._focus_plotter(ax3, data[:, self.delta], x, ylim[0], ylim[1], height)
        ax1.set_title(r"Axis of Focus")
        ax2.set_title(r"Focal Plane, varying x")
        ax3.set_title(r"Focal Plane, varying y")
        fig.suptitle(r"Slices through the focus.")
        fig.show()

    @staticmethod
    def _focus_plotter(ax, data, x, xmin, xmax, height):
        ax.plot(x * 1e6, data)
        ax.axvline(xmin * 1e6, color="red", linestyle="dotted")
        ax.axvline(xmax * 1e6, color="red", linestyle="dotted")
        ax.axhline(height, color="red", linestyle="dotted")
        ax.set_xlabel(r"Relative position ($\mu$m)")
        ax.set_ylabel(r"Counts/s")

    def characterise_save(self, plot=False, filename="Characteristics_focus"):
        text = self.characterise(plot)
        fullfilename = sf.find_free_filename(self.savepath, filename, ".txt")
        with open(fullfilename, "w") as f:
            f.write(text)

    @staticmethod
    def _focus_char_string(power, area, waist, z_R):
        text = ""
        for i in range(len(power)):
            text += "CCD {}:\n".format(i)
            text += "Total power: {} counts in area {} m^2\n".format(power[i], area[i])
            text += "x-waist: {} m, y-waist: {} m, mean: {}m\n".format(*waist[i][1:], np.mean(waist[i][1:]))
            circ = np.pi * np.mean(waist[i][1:])**2
            text += "Perfect circle: {} m^2, area/circle: {}\n".format(circ, area[i] / circ)
            text += "The Rayleigh ranges: {}, {} m\n".format(*z_R[i])
            text += "z-waist: {} m\n\n".format(waist[i][0])
        return text

    def findF(self, I0=5.88052e9):
        F = []
        Omega = []
        for i in self.focus:
            z0 = int(np.argwhere(i[:, self.delta, self.delta] == np.max(i[:, self.delta, self.delta])).reshape(-1)[0])
            x = np.arange(-self.delta, self.delta + 1) * self.pixel_size
            ff = []
            data = i[z0, :, :]
            height = data[self.delta, self.delta] * np.exp(-2)
            w1, _ = self._find_waist(data[self.delta, :], x, height)
            w2, _ = self._find_waist(data[:, self.delta], x, height)
            waist = np.mean([w1, w2])
            focalpower, focalarea = self._focalpower(data, waist, True)
            Omega.append((focalpower / focalarea) / I0)
            for j in range(len(self.z)):
                power, area = np.sum(i[j, :, :]), 1
                if area > 0:
                    ff.append(power)
                else:
                    ff.append(np.nan)
            ff = np.array(ff)
            F.append(focalpower / ff)
        return F, Omega
