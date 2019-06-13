from CCDFocus import CCDFocus
from errors import FocusGeneratorError

from scipy import special as special
import numpy as np


class FocusGenerator(CCDFocus):

    def __init__(self, rho, z, wavelength, A, B, theta_open=np.pi/4, I0=1):
        rho_norm, z_norm = self.normalize_rho_z(rho, z, wavelength)
        self.cubed = np.array([self.aberrated_focus(rho_norm, z_norm, A, B, theta_open, I0)])
        self.z = z_norm
        self.delta = int(len(rho) / 2)
        self.pixel_size = wavelength

    @staticmethod
    def _error(exception=None):
        """Raises the FocusGeneratorError."""
        return FocusGeneratorError(exception)


    @staticmethod
    def normalize_rho_z(rho, z, wavelength):
        """'Normalize' rho and z using the opening angle and
        the wavelength of the laser.
        """
        k = 2 * np.pi / wavelength
        rho_norm = rho * k
        z_norm = z * k
        return rho_norm, z_norm

    @staticmethod
    def aberrated_focus(rho_norm, z_norm, A, B, theta_max, I0):
        """Given a list of normalized rho and z, as well as the
        aberration angles thata ([theta_max, theta_sphere, theta_gaus])
        and power I0, calculate the aberrated focus of a laser.
        This is done using the following integral:
        E = E0*int[2 * q * exp{i * (pi * sqrt(1-q^2) * z)
            - pi * (q / q_sphere)^4 - (q / q_gaus)^2}
            * J0(pi*rho*q)]dq ; from 0 to q_max
        with J0 the zeroth order bessel function of the first kind.
        The intensity is simply |E|^2.
        """

        rho, z = np.array(rho_norm), np.array(z_norm)
        q = np.linspace(0, np.sin(theta_max), 1000)  # integration constant

        rho_2D = np.sqrt(np.sum(np.power(np.meshgrid(rho, rho), 2), axis=0))  # rho_2D[i, j] = sqrt(rho[i]^2 + rho[j]^2)
        J0 = special.j0(np.pi * np.outer(rho_2D, q).reshape(*rho_2D.shape, *q.shape))
        # J0[i, j, k] = J0(sqrt(rho[i]^2 + rho[j]^2) * q[k])

        z_2D = np.outer(z, np.sqrt(1 - q**2))
        q_2D = np.outer(np.ones(len(z)), q)
        expphi = np.array(2 * q_2D * np.exp(-B * q_2D**2 - 1j * (np.pi * A * q_2D**4 - z_2D)))

        E = np.inner(expphi, J0) / len(q)  #"Integrate" over the function, giving a 3d array (matrix mult.)
        # E is a 3D matrix which looks like E[z, rho, rho]
        return I0 * np.abs(E)**2
