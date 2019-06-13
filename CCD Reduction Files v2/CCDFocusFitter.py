from CCDFocus import CCDFocus
from FocusGenerator import FocusGenerator
from errors import CCDFocusFitterError

import numpy as np
from matplotlib import pyplot as plt
import emcee  # make sure they are installed. Use conda install -c astropy emcee to install from Anaconda Prompt
import corner  # use conda install -c astropy corner


class CCDFocusFitter(CCDFocus):

    def __init__(self, folderpath, masterpath=None, savepath=None, delta=150, pixel_size=9e-6,
                 magnification=100, realign=False, wavelength=520e-9, theta_open=np.pi/4):
        super(CCDFocusFitter, self).__init__(folderpath, masterpath, savepath, delta, pixel_size,
                                             magnification, realign)
        self.wavelength = wavelength
        self.theta_open = theta_open


    @staticmethod
    def _error(exception=None):
        """Raises the CCDFocusFitterError."""
        return CCDFocusFitterError(exception)


    def fitter(self, nwalkers=8, cutoff=70, alpha=0.16, plot=True):
        ndim = 4  # A, B, z0, offset
        theta_best = theta_sigma = [], []
        for i in self.cubed:
            print("1!")
            sampler = self._create_sampler(i, ndim, nwalkers)
            print("2!")
            if plot is True:
                self._sample_plotter(sampler, cutoff, alpha, ndim)
            print("3!")
            samples = sampler.chain[:, cutoff:, :].reshape((-1, ndim))
            print("4!")
            t_best, t_sigma = self._calc_intervals(samples, alpha)
            theta_best.append(t_best)
            theta_sigma.append(t_sigma)
        return theta_best, theta_sigma

    def _create_sampler(self, cube, ndim, nwalkers):
        rho = np.arange(-self.delta, self.delta + 1) * self.pixel_size
        theta_guess = np.array([0, 5, 0, 0])  # A, B, z0, offset
        theta_width = np.array([100, 5, len(self.z), 5 * np.mean(cube)])

        pos = theta_guess * np.ones((nwalkers, ndim)) + 1e-1 * theta_width * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.Probability, args=(theta_guess, theta_width, rho,
                                                                                self.z, cube))
        sampler.run_mcmc(pos, 500)
        return sampler

    def Probability(self, theta, theta_guess, theta_width, rho, z, focus_array):
        print("Probability!!")
        return self.Prior(theta, theta_guess, theta_width) + self.lnL(theta, rho, z, focus_array)

    def lnL(self, theta, rho, z, focus_array):
        th = [self.wavelength, *theta[:2], self.theta_open, 1]
        model = FocusGenerator(rho, z-theta[2], *th).cubed[0] + theta[3]
        m = model / np.sum(model) * np.sum(focus_array)

        return -0.5 * np.sum((focus_array - m)**2)

    @staticmethod
    def Prior(theta, theta_guess, theta_width):
        return np.sum(-(theta - theta_guess)**2 / (2 * theta_width**2))

    @staticmethod
    def _sample_plotter(sampler, cutoff, alpha, ndim):
        fig, axes = plt.subplots(ncols=1, nrows=4)
        fig.set_size_inches(12, 6)
        fig.suptitle("Walker evolution of the various parameters")
        for i in range(3):
            axes[i].plot(sampler.chain[:, :, i].transpose(), color='black', alpha=0.3)
            axes[i].axvline(cutoff, ls='dashed', color='red')
        axes[0].set_ylabel('A')
        axes[1].set_ylabel('B')
        axes[2].set_ylabel('z0')
        axes[3].set_ylabel('offset')
        axes[3].set_xlabel("Walker steps")
        fig.show()

        samples = sampler.chain[:, cutoff:, :].reshape((-1, ndim))
        fig = corner.corner(samples, bins=30, labels=['A','B','I0','z0', 'offset'], quantiles=[alpha, 0.5, 1 - alpha])
        fig.suptitle("Corner plot of the parameters.")
        fig.show()

    @staticmethod
    def _calc_intervals(samples, alpha):
        theta_best = []
        theta_sigma = []

        for i in range(len(samples[0, :])):
            theta_best.append(np.percentile(samples[:, i], 50))
            theta_sigma.append(0.5 * (np.quantile(samples[:,i], 1 - alpha) - np.quantile(samples[:,i], alpha)))
        return np.array(theta_best), np.array(theta_sigma)
