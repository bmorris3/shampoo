from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from scipy.optimize import fmin

from .lorenzmie import lorenz_mie_field_cartesian_suppressed


__all__ = ['LorenzMieModel']


class LorenzMieModel(object):
    def __init__(self, hologram, pixel_size, wavelength=0.405,
                 n_medium=1.33):
        """
        Parameters
        ----------
        hologram : `~numpy.ndarray`
            Hologram to fit

        initial_parameters : `~numpy.ndarray` or list
            Initial fit parameters

        wavelength : float
            Laser wavelength in microns

        n_medium : float
            Index of refraction of the medium (water)
        """
        # Normalize the hologram
        self.hologram = (hologram - np.median(hologram))/np.std(hologram)
        self.wavelength = wavelength
        self.n_medium = n_medium
        self.best_fit_parameters = None
        self.best_fit_model = None
        self.pixel_size = pixel_size

    def model_holo(self, p, xy_centroid):
        z, radius, amp, n_sphere, exp_amp = p
        x, y = xy_centroid[1], xy_centroid[0]
        pos = np.array([x, y, z])
        holo = lorenz_mie_field_cartesian_suppressed(pos, radius, n_sphere,
                                                     self.n_medium,
                                                     self.wavelength,
                                                     self.pixel_size,
                                                     list(self.hologram.shape),
                                                     exp_amp) * amp
        return holo

    def fit(self, initial_parameters, parameter_bounds, xy_centroid):
        """
        Fit the Lorenz-Mie scattering model to the hologram.

        Parameters include: ``z, radius, amp, n_sphere, exp_amp``

        Parameters
        ----------
        initial_parameters : list
        parameter_bounds : list of tuples
        xy_centroid : list

        """
        results = fmin(lambda p, xy_centroid: np.sum(np.abs(self.hologram -
                                                            self.model_holo(p, xy_centroid))),
                            initial_parameters,
                            args=(xy_centroid,))
                            #bounds=parameter_bounds, approx_grad=True

        self.best_fit_parameters = results
        self.best_fit_model = self.model_holo(self.best_fit_parameters, xy_centroid)
