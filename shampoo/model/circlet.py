from multiprocessing.dummy import Pool as ThreadPool
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2

import numpy as np

__all__ = ['CircletTransform']

class CircletTransform(object):
    """
    Circlet transform as defined by Chauris et al. 2011 [1]_

    .. [1] http://archimer.ifremer.fr/doc/00033/14451/11752.pdf
    """
    def __init__(self, image, k=2, N=3, threads=8):
        self.k = k
        self.N = N
        self.omegas = np.linspace(-np.pi, np.pi, image.shape[0])
        self.omega_k = np.pi * (self.k - 1)/(self.N - 1)
        self.fft_image = fft2(image)
        self._F_k = None

        freqs = np.fft.fftfreq(self.fft_image.shape[0])
        w_1, w_2 = np.meshgrid(2*np.pi*freqs, 2*np.pi*freqs)
        self.mag_w = np.sqrt(w_1**2 + w_2**2)
        self.threads = threads

    @property
    def F_k(self):
        # Cache this value
        if self._F_k is None:
            result = np.zeros_like(self.omegas)
            in_bounds_plus = (np.abs(self.omegas + self.omega_k) <= np.pi/(self.N-1))
            in_bounds_minus = (np.abs(self.omegas - self.omega_k) <= np.pi/(self.N-1))
            result[in_bounds_plus] = np.cos(self.omegas[in_bounds_plus] + self.omega_k)
            result[in_bounds_minus] = np.cos(self.omegas[in_bounds_minus] - self.omega_k)

            # Normalize result, Chauris et al. 2011, Equation 5
            self._F_k = result/np.sum(np.abs(result)**2)

        return self._F_k

    def G_k(self, radius):
        # Chauris et al. 2011, Equation 7:
        result = np.exp(1j * self.mag_w * radius) * self.F_k

        # Normalize result, Chauris et al. 2011, Equation 6
        return result/np.sum(np.abs(result)**2)

    def coefficients(self, radius):
        circlet_coefficients = ifft2(self.fft_image*self.G_k(radius))
        return circlet_coefficients

    def coefficients_multithread(self, radii):

        _ = self.F_k
        circlet_coefficients = np.zeros((len(radii), self.fft_image.shape[0],
                                         self.fft_image.shape[1]))
        def job(i):
            circlet_coefficients[i, ...] = self.coefficients(radii[i])

        pool = ThreadPool(self.threads)
        pool.map(job, range(len(radii)))

        # close the pool and wait for the work to finish
        pool.close()
        pool.join()
        return circlet_coefficients

    def reconstruct(self, coefficients, radius):
        reconstructed_image = ifft2(fft2(coefficients)*np.conj(self.G_k(radius)))
        return reconstructed_image
