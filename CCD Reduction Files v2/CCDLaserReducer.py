import support_functions as sf
from CCDReducer import CCDReducer
from errors import CCDLaserReducerError

from scipy import optimize
from matplotlib import pyplot as plt, colors
import numpy as np


class CCDLaserReducer(CCDReducer):
    """Loads a single file and reduces it. Has special functions specific for lasers."""

    def __init__(self, filepath, masterpath=None, savepath=None, pixel_size=1e-6, magnification=1):
        super(CCDLaserReducer, self).__init__(filepath, masterpath, savepath)
        try:
            assert isinstance(pixel_size, (float, int)), "pixel_size must be a float or string."
            self.pixel_size = pixel_size/magnification
        except AssertionError as excep:
            raise self._error(excep) from excep

    @staticmethod
    def _error(exception=None):
        """Raises the CCDLaserReducerError."""
        return CCDLaserReducerError(exception)

    def _imshow(self, fig, ax, data, cmap, log):
        ps = self.pixel_size * 1e6
        extent = [-0.5 * ps, (data.shape[1] - 0.5) * ps, (data.shape[0] - 0.5) * ps, -0.5 * ps]
        if log is False:
            im = ax.imshow(data, cmap=cmap, aspect="equal", extent=extent)
        else:
            im = ax.imshow(data, cmap=cmap, aspect="equal", extent=extent, norm=colors.LogNorm())
        ax.set_xlabel(r"x position ($\mu$m)")
        ax.set_ylabel(r"y position ($\mu$m)")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Counts")

    def sliceshow(self, positions=[None, None], title="Slice of Data", fit=False, log=False):
        try:
            assert isinstance(title, str), "title must be a string"
        except AssertionError as excep:
            raise self._error(excep) from excep

        best_pos = self._find_slice_pos(positions)
        for i in range(len(self.data)):
            fig = plt.figure(figsize=[15, 10.5])
            self._slicing(fig, self.data.data()[i], self.data.time()[i], best_pos[i], fit, log)
            if i == 0:
                fig.suptitle(title)
            else:
                fig.suptitle(title + " " + str(i))

            fig.show()

    def _slicing(self, fig, data, time, position, fit, log):
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(212)
        self._imshow(fig, ax3, data, "jet", log)
        ax3.set_title("Exposure Time = "+str(round(time, 6)))
        ps = self.pixel_size * 1e6
        ax3.axhline(position[0] * ps, color="blue")
        ax3.axvline(position[1] * ps, color="red")
        if log is not False:
            ax1.semilogy()
            ax2.semilogy()

        self._slice1D(ax1, data[position[0], :], fit, ps, "blue")
        ax1.set_title("y position:" + str(np.round(position[0] * ps, 3)) + r" $\mu$m")
        ax1.set_xlabel(r"x position ($\mu$m)")
        ax1.set_ylabel("Counts")

        self._slice1D(ax2, data[:, position[1]], fit, ps, "red")
        ax2.set_title("x-position:" + str(np.round(position[1] * ps, 3)) + r" $mu$m")
        ax2.set_xlabel(r"y position ($\mu$m)")

    def _find_slice_pos(self, positions):
        length = len(self.data)
        y, x = positions
        best_pos = self.find_laser().transpose()

        if x is not None:
            self._check_integer_filled_list(x, "x", length)
        else:
            x = best_pos[1]

        if y is not None:
            self._check_integer_filled_list(y, "y", length)
        else:
            y = best_pos[0]

        return np.array([y, x]).transpose()

    def _check_integer_filled_list(self, x, name, size=None):
        name = str(name)
        if not isinstance(x, list):
            raise self._error(str(name) + " must be a list filled with integers")
        if size is not None:
            size = int(size)
            if len(x) != size:
                raise self._error(str(name) + " must have the following size:" + str(size))
        for i in x:
            if not isinstance(i, int):
                raise self._error(str(name) + " must be filled with integers")

    def find_laser(self):
        data = self.data.data()
        maxima = []
        for i in data:
            maxima.append(np.argwhere(i == np.max(i))[0])
        return np.array(maxima)

    def slicesave(self, position=[None, None], title="Slice of Data", fit=False, log=False, savename="Slice",
                        extension=".png"):
        try:
            assert isinstance(title, str), "title must be a string"
            assert isinstance(savename, str), "savename must be a string"
            assert isinstance(extension, str), "extension must be a string"
        except AssertionError as excep:
            raise self._error(excep) from excep

        best_pos = self._find_slice_pos(position)
        for i in range(len(self.data)):
            fig = plt.figure(figsize=[15, 10.5])
            self._slicing(fig, self.data.data()[i], self.data.time()[i], best_pos[i], fit, log)
            if i == 0:
                fig.suptitle(title)
            else:
                fig.suptitle(title + " " + str(i))

            full_filename = sf.find_free_filename(self.savepath, savename, extension)
            fig.savefig(full_filename)
            plt.close(fig)

    def _slice1D(self, ax, sliced, fit, ps, color):
        x = np.arange(len(sliced)) * ps
        ax.plot(x, sliced, color=color)
        if fit is True:
            self._add_fit_to_axis(ax, sliced, x, color)

    def _add_fit_to_axis(self, ax, fitdata, x, color):
        popt, success = self._gaussian_fit(fitdata, x)
        y = np.linspace(np.min(x), np.max(x), 1000)
        if success is False:
            print("Optimal parameters not found")
        labelstring = self._legend_fit_values(popt, success)
        ax.plot(y, sf.gauss(y, *popt, 0), color=color, linestyle="dotted", label=labelstring)
        ax.legend(loc="best", fontsize=10)

    @staticmethod
    def _gaussian_fit(data, x):
        x0 = x[np.argwhere(data == np.max(data))[0][0]]
        sigma = np.sqrt(sf.moment(x, 2, data))
        init_guess = [np.max(data), sigma, x0]
        bounds = [[np.min(data), 0, np.min(x)], [np.inf, np.inf, np.max(x)]]
        def f(x, A, sig, x0pos): return sf.gauss(x, A, sig, x0pos, 0)
        try:
            popt, _ = optimize.curve_fit(f, x, data, p0=init_guess, bounds=bounds)
            return popt, True
        except RuntimeError:
            return init_guess, False

    @staticmethod
    def _legend_fit_values(popt, success):
        A = np.round(popt[0])
        sigma = np.round(popt[1], 5)
        x0 = np.round(popt[2], 5)
        labelstring = "Fit values:\n" + r"$A=${}".format(A) + "\n" + r"$\sigma=${}".format(sigma) + "\n" +\
            r"$x_0={}$".format(x0)
        if success is False:
            labelstring += "\n fit dit not succeed."
        return labelstring

    def powershow(self, title="Image of Data"):
        try:
            assert isinstance(title, str), "title must be a string"
        except AssertionError as excep:
            raise self._error(excep) from excep

        p = self.cum_power_fraction_within_area()
        for i in range(len(p)):
            fig = self._power(p[i])
            if i == 0:
                fig.suptitle(title)
            else:
                fig.suptitle(title + " " + i)
            fig.show()

    def powersave(self, title="Image of Data", savename="FitsPicture", extension=".png"):
        try:
            assert isinstance(title, str), "title must be a string"
            assert isinstance(savename, str), "savename must be a string"
            assert isinstance(extension, str), "extension must be a string"
        except AssertionError as excep:
            raise self._error(excep) from excep

        p = self.cum_power_fraction_within_area()
        for i in range(len(p)):
            full_filename = sf.find_free_filename(self.savepath, savename, extension)
            fig = self._power(p[i])
            if i == 0:
                fig.suptitle(title)
            else:
                fig.suptitle(title + " " + i)
            fig.savefig(full_filename)
            plt.close(fig)

    def _power(self, powerlist):
        powerlist = np.array(powerlist)
        x = np.arange(len(powerlist)) * self.pixel_size * 1e3
        fig = plt.figure(figsize=[15, 10.5])
        ax = fig.add_subplot(111)
        ax.plot(x, powerlist * 100)
        ax.set_xlabel("Distance from center (mm)")
        ax.set_ylabel("Percentage of total power on detector")
        return fig

    def cum_power_fraction_within_area(self):
        data = self.data.data()
        best_pos = self.find_laser()
        distance = self._smallest_distance_to_edge(best_pos)
        power = []
        for i in range(len(data)):
            power2 = []
            for j in range(distance[i]):
                p = np.sum(data[i] * sf.aperturized(data[i].shape, best_pos[i], j))
                power2.append(p / np.sum(data[i]))
            power.append(np.array(power2))
        return power

    def _smallest_distance_to_edge(self, position):
        data = self.data.data()
        assert len(data) == len(position), "position length must be equal to data length"
        distances = []
        for i in range(len(position)):
            edge_x_distance = np.min([position[i][1], data[i].shape[1]-position[i][1]])
            edge_y_distance = np.min([position[i][0], data[i].shape[0]-position[i][0]])
            distances.append(np.min([edge_x_distance, edge_y_distance]))
        return distances

    def power_within_area(self, radius, position=None):
        data = self.data.data()
        if position is None:
            position = self.find_laser()
        distance = self._smallest_distance_to_edge(position)
        if radius + 0.5 >= np.min(distance):
            print("Part of the aperture lies outside of the dataset")
        power = []
        for i in range(len(data)):
            p = np.sum(data[i] * sf.aperturized(data[i].shape, position[i], radius))
            power.append(p)
        return power
