"""
This module handles reconstruction of phase and intensity images from raw
holograms using "the convolution approach": see Section 3.3 of Schnars & Juptner
(2002) Meas. Sci. Technol. 13 R85-R101 [1]_.

Aberration corrections from Colomb et al., Appl Opt. 2006 Feb 10;45(5):851-63
are applied [2]_.

    .. [1] http://x-ray.ucsd.edu/mediawiki/images/d/df/Digital_recording_numerical_reconstruction.pdf
    .. [2] http://www.ncbi.nlm.nih.gov/pubmed/16512526

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.ndimage import gaussian_filter, filters
from skimage.restoration import unwrap_phase
from skimage.io import imread

# Use the 'agg' backend if on Linux
import sys
import matplotlib
import matplotlib.pyplot as plt
if 'linux' in sys.platform:
    matplotlib.use('agg')

try:
   from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
except ImportError:
    from scipy.fftpack import fft2, ifft2

__all__ = ['Hologram', 'ReconstructedWavefield']


RANDOM_SEED = 42


def rebin_image(a, binning_factor):
    # Courtesy of J.F. Sebastian: http://stackoverflow.com/a/8090605
    if binning_factor == 1:
        return a

    new_shape = (a.shape[0]/binning_factor, a.shape[1]/binning_factor)
    sh = (new_shape[0], a.shape[0]//new_shape[0], new_shape[1],
          a.shape[1]//new_shape[1])
    return a.reshape(sh).mean(-1).mean(1)


def shift_peak(arr, shifts_xy):
    """
    2D array shifter.

    Take 2D array `arr` and "roll" the contents in two dimensions, with shift
    magnitudes set by the elements of `shifts` for the ``x`` and ``y``
    directions respectively.

    Parameters
    ----------
    arr : ndarray with dimensions ``N`` x ``M``
        Array to shift
    shifts_xy : list of length ``M``
        Desired shifts in ``x`` and ``y`` directions respectively

    Returns
    -------
    shifted_arr : ndarray
        Input array with elements shifted ``shifts_xy[0]`` pixels in ``x`` and
        ``shifts_xy[1]`` pixels in ``y``.
    """
    return np.roll(np.roll(arr, int(shifts_xy[0]), axis=0),
                   int(shifts_xy[1]), axis=1)


def make_items_hashable(input_iterable):
    """
    Take a list or tuple of objects, convert any items that are lists into
    tuples to make them hashable.

    Parameters
    ----------
    input_iterable : list or tuple
        Items to convert to hashable types

    Returns
    -------
    hashable_tuple : tuple
        ``input_iterable`` made hashable
    """
    return tuple([tuple(i) if isinstance(i, list) or isinstance(i, np.ndarray)
                  else i for i in input_iterable])


def _load_hologram(hologram_path):
    """
    Load a hologram from path ``hologram_path`` using scikit-image and numpy.
    """
    return np.array(imread(hologram_path), dtype=np.float64)


class Hologram(object):
    """
    Container for holograms and methods to reconstruct them.
    """
    def __init__(self, hologram, wavelength=405e-9,
                 rebin_factor=1, dx=3.45e-6, dy=3.45e-6):
        """
        Parameters
        ----------
        hologram : `~numpy.ndarray`
            Input hologram
        wavelength : float [meters]
            Wavelength of laser
        rebin_factor : int
            Rebin the image by factor ``rebin_factor``. Must be an even integer.
        dx : float [meters]
            Pixel width in x-direction (unbinned)
        dy: float [meters]
            Pixel width in y-direction (unbinned)
        """

        self.rebin_factor = rebin_factor
        self.hologram = rebin_image(np.float64(hologram), self.rebin_factor)
        self.n = self.hologram.shape[0]
        self.wavelength = wavelength
        self.wavenumber = 2*np.pi/self.wavelength
        self.reconstructions = dict()
        self.dx = dx*rebin_factor
        self.dy = dy*rebin_factor
        self.mgrid = np.mgrid[0:self.n, 0:self.n]
        self.random_seed = RANDOM_SEED
        self.digital_phase_mask = None

    @classmethod
    def from_tif(cls, hologram_path, **kwargs):
        """
        Open a raw hologram from a tif file.

        Parameters
        ----------
        hologram_path : string
            Path to input hologram to reconstruct

        kwargs : dict
            Keyword arguments to pass to the `~shampoo.reconstruction.Hologram`
            constructor.

        """
        hologram = _load_hologram(hologram_path)
        return cls(hologram, **kwargs)

    def reconstruct(self, propagation_distance,
                    plot_aberration_correction=False,
                    plot_fourier_peak=False,
                    cache=False, digital_phase_mask=None):
        """
        Wrapper around `~shampoo.reconstruction.reconstruct_wavefield` for
        caching.

        Parameters
        ----------
        propagation_distance : float
            Propagation distance [m]
        plot_aberration_correction : bool
            Plot the abberation correction visualization? Default is False.
        plot_fourier_peak : bool
            Plot the peak-centroiding visualization of the fourier transform
            of the hologram? Default is False.
        cache : bool
            Cache reconstructions onto the hologram object? Default is False.
        digital_phase_mask : `~numpy.ndarray`
            Digital phase mask, if you have one precomputed. Default is None.

        Returns
        -------
        reconstructed_wavefield : `~shampoo.reconstruction.ReconstructedWavefield`
            The reconstructed wavefield.
        """
        self.digital_phase_mask = digital_phase_mask

        if cache:
            cache_key = make_items_hashable((propagation_distance,
                                             self.wavelength, self.dx, self.dy))

        # If this reconstruction is cached, get it.
        if cache and cache_key in self.reconstructions:
            reconstructed_wavefield = self.reconstructions[cache_key]

        # If this reconstruction is not in the cache,
        # or if the cache is turned off, do the reconstruction
        elif (cache and cache_key not in self.reconstructions) or not cache:
            reconstructed_wavefield  = self.reconstruct_wavefield(propagation_distance,
                                                                  plot_aberration_correction=plot_aberration_correction,
                                                                  plot_fourier_peak=plot_fourier_peak)

        # If this reconstruction should be cached and it is not:
        if cache and cache_key not in self.reconstructions:
            self.reconstructions[cache_key] = ReconstructedWavefield(reconstructed_wavefield)

        return ReconstructedWavefield(reconstructed_wavefield)

    def reconstruct_wavefield(self, propagation_distance,
                              plot_aberration_correction=False,
                              plot_fourier_peak=False):
        """
        Reconstruct wavefield from hologram stored in file ``hologram_path`` at
        propagation distance ``propagation_distance``.

        Parameters
        ----------
        propagation_distance : float
            Propagation distance [m]
        plot_aberration_correction : bool
            Plot the abberation correction visualization? Default is False.
        plot_fourier_peak : bool
            Plot the peak-centroiding visualization of the fourier transform
            of the hologram? Default is False.

        Returns
        -------
        reconstructed_wavefield : `~numpy.ndarray` (complex)
            Reconstructed wavefield from hologram
        """
        # Read input image
        apodized_hologram = self.apodize(self.hologram)

        # Isolate the real image in Fourier space, find spectral peak
        F_hologram = fft2(apodized_hologram)
        spectrum_centroid = self.find_fourier_peak_centroid(F_hologram,
                                                            plot=plot_fourier_peak)

        # Create mask based on coords of spectral peak:
        mask_radius = 150./self.rebin_factor
        mask = self.generate_real_image_mask(spectrum_centroid[0],
                                             spectrum_centroid[1],
                                             mask_radius)

        # Calculate Fourier transform of impulse response function
        G = self.fourier_trans_of_impulse_resp_func(propagation_distance)

        # plt.imshow(np.log(np.abs(G)), interpolation='nearest')
        # #plt.plot(spectrum_centroid[0], spectrum_centroid[1], 'o')
        # plt.plot(self.n/2, self.n/2, 'bo')
        # plt.title('G')
        # plt.show()

        if self.digital_phase_mask is None:
            # Center the spectral peak
            shifted_F_hologram = shift_peak(F_hologram*mask,
                                            [self.n/2-spectrum_centroid[1],
                                             self.n/2-spectrum_centroid[0]])

            # Apodize the result
            psi = self.apodize(shifted_F_hologram*G)
            # psi = shifted_F_hologram*G

            # Calculate reference wave
            self.digital_phase_mask = self.get_digital_phase_mask(psi,
                                                                  plots=plot_aberration_correction)

        # Reconstruct the image
        psi = G*shift_peak(fft2(apodized_hologram * self.digital_phase_mask) * mask,
                           [self.n/2 - spectrum_centroid[1],
                            self.n/2 - spectrum_centroid[0]])

        reconstructed_wavefield = shift_peak(ifft2(psi),
                                             [self.n/2, self.n/2])
        return reconstructed_wavefield

    def get_digital_phase_mask(self, psi, plots=False):
        """
        Calculate the digital phase mask (i.e. reference wave), as in Colomb et
        al. 2006, Eqn. 26 [1]_.

        Fit for a second order polynomial, numerical parametric lens with least
        squares to remove tilt, spherical aberration.

        .. [1] http://www.ncbi.nlm.nih.gov/pubmed/16512526

        Parameters
        ----------
        psi : `~numpy.ndarray`
            The product of the Fourier transform of the hologram and the Fourier
            transform of impulse response function
        plots : bool
            Display plots after calculation if `True`

        Returns
        -------
        phase_mask : `~numpy.ndarray`
            Digital phase mask, used for correcting phase aberrations.
        """
        y, x = self.mgrid - self.n/2

        inverse_psi = shift_peak(ifft2(psi), [self.n/2, self.n/2])

        phase_image = np.arctan(np.imag(inverse_psi)/np.real(inverse_psi))
        unwrapped_phase_image = unwrap_phase(2*phase_image,
                                             seed=self.random_seed)/2/self.wavenumber
        smooth_phase_image = gaussian_filter(unwrapped_phase_image, 50)

        # Fit the smoothed phase image with a 2nd order polynomial surface with
        # mixed terms using least-squares.
        v = np.array([np.ones(len(x[0, :])), x[0, :], y[:, 0], x[0, :]**2,
                      x[0, :] * y[:, 0], y[:, 0]**2])
        coefficients = np.linalg.lstsq(v.T, smooth_phase_image)[0]
        field_curvature_mask = np.dot(v.T, coefficients)

        digital_phase_mask = np.exp(-1j*self.wavenumber * field_curvature_mask)

        if plots:
            fig, ax = plt.subplots(1, 2, figsize=(14, 8))
            ax[0].imshow(unwrapped_phase_image, origin='lower')
            ax[1].imshow(field_curvature_mask, origin='lower')
            plt.show()

        return digital_phase_mask

    def apodize(self, arr):
        """
        Force the magnitude of an array to go to zero at the boundaries.

        Parameters
        ----------
        arr : `~numpy.ndarray`
            Array to apodize

        Returns
        -------
        apodized_arr : `~numpy.ndarray`
            Apodized array
        """
        y, x = self.mgrid
        arr *= (np.sqrt(np.cos((x-self.n/2.)*np.pi/self.n)) *
                np.sqrt(np.cos((y-self.n/2.)*np.pi/self.n)))
        return arr

    def fourier_trans_of_impulse_resp_func(self, propagation_distance):
        """
        Calculate the Fourier transform of impulse response function, sometimes
        represented as ``G``.

        For reference, see Eqn 3.22 of Schnars & Juptner (2002) Meas. Sci.
        Technol. 13 R85-R101 [1]_.

        .. [1] http://x-ray.ucsd.edu/mediawiki/images/d/df/Digital_recording_numerical_reconstruction.pdf

        Parameters
        ----------
        propagation_distance : float
            Propagation distance [m]

        Returns
        -------
        G : `~numpy.ndarray`
            Fourier transform of impulse response function
        """
        y, x = self.mgrid - self.n/2
        first_term = (self.wavelength**2 * (x + self.n**2 * self.dx**2 /
                      (2.0 * propagation_distance * self.wavelength))**2 /
                      (self.n**2 * self.dx**2))
        second_term = (self.wavelength**2 * (y + self.n**2 * self.dy**2 /
                       (2.0 * propagation_distance * self.wavelength))**2 /
                       (self.n**2 * self.dy**2))
        G = np.exp(-1j * self.wavenumber* propagation_distance *
                   np.sqrt(1.0 - first_term - second_term))
        return G

    def generate_real_image_mask(self, center_x, center_y, radius):
        """
        Calculate the Fourier-space mask to isolate the real image
    
        Parameters
        ----------
        center_x : int
            ``x`` centroid [pixels] of real image in Fourier space
        center_y : int
            ``y`` centroid [pixels] of real image in Fourier space
        radius : float
            Radial width of mask [pixels] to apply to the real image in Fourier
            space
    
        Returns
        -------
        mask : `~numpy.ndarray`
            Binary-valued mask centered on the real-image peak in the Fourier
            transform of the hologram.
        """
        y, x = self.mgrid
        mask = np.zeros((self.n, self.n))
        mask[(x-center_x)**2 + (y-center_y)**2 < radius**2] = 1.0
        return mask
    
    def find_fourier_peak_centroid(self, fourier_arr, margin_factor=0.1,
                                   plot=False):
        """
        Calculate the centroid of the signal spike in Fourier space near the
        frequencies of the real image.

        Parameters
        ----------
        fourier_arr : `~numpy.ndarray`
            Fourier-transform of the hologram
        margin_factor : int
            Fraction of the length of the Fourier-transform of the hologram
            to ignore near the edges, where spurious peaks occur there.
        plot : bool
            Plot the peak-centroiding visualization of the fourier transform
            of the hologram? Default is False.

        Returns
        -------
        pixel : `~numpy.ndarray`
            Pixel at the centroid of the spike in Fourier transform of the
            hologram near the real image.
        """
        margin = int(self.n*margin_factor)
        abs_fourier_arr = filters.gaussian_filter(np.abs(fourier_arr)[margin:-margin,
                                                  margin:-margin], 10)
                                                  #margin:-margin], 2)
        spectrum_centroid = np.array(np.unravel_index(abs_fourier_arr.T.argmax(),
                                     abs_fourier_arr.shape)) + margin

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(18, 6))

            ax[0].imshow(abs_fourier_arr, interpolation='nearest',
                        origin='lower')
            ax[0].plot(spectrum_centroid[0]-margin, spectrum_centroid[1]-margin, 'o')

            ax[1].imshow(np.log(np.abs(fourier_arr)), interpolation='nearest',
                        origin='lower')
            ax[1].plot(spectrum_centroid[0], spectrum_centroid[1], 'o')
            plt.show()
        return spectrum_centroid


class ReconstructedWavefield(object):
    """
    Container for reconstructed wavefields and their intensity and phase
    arrays.
    """
    def __init__(self, reconstructed_wavefield):
        self.reconstructed_wavefield = reconstructed_wavefield
        self._intensity_image = None
        self._phase_image = None
        self.random_seed = RANDOM_SEED

    @property
    def intensity(self):
        """
        `~numpy.ndarray` of the reconstructed intensity.
        """
        if self._intensity_image is None:
            self._intensity_image = np.abs(self.reconstructed_wavefield)
        return self._intensity_image

    @property
    def phase(self):
        """
        `~numpy.ndarray` of the reconstructed phase. The phase has been
        unwrapped.
        """
        if self._phase_image is None:
            self._phase_image = unwrap_phase(2*np.arctan(np.imag(self.reconstructed_wavefield)/
                                             np.real(self.reconstructed_wavefield)),
                                             seed=self.random_seed)

        return self._phase_image

    def plot(self, phase=False, intensity=False, all=None, cmap=plt.cm.binary_r):
        """

        Parameters
        ----------
        phase:
        intensity:
        all:
        cmap:

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
        ax : `~matplotlib.axes.Axes`
        """
        phase_kwargs = dict(vmin=np.percentile(self.phase, 0.1),
                            vmax=np.percentile(self.phase, 99.9))

        if all is None:
            if phase and not intensity:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(self.phase[::-1, ::-1], cmap=cmap,
                          origin='lower', interpolation='nearest',
                          **phase_kwargs)
            elif intensity and not phase:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(self.intensity[::-1, ::-1], cmap=cmap,
                          origin='lower', interpolation='nearest')
        else:
            fig, ax = plt.subplots(1, 2, figsize=(18,8), sharex=True, sharey=True)
            ax[0].imshow(self.intensity[::-1, ::-1], cmap=cmap,
                         origin='lower', interpolation='nearest')
            ax[0].set(title='Intensity')
            ax[1].imshow(self.phase[::-1, ::-1], cmap=cmap,
                         origin='lower', interpolation='nearest',
                         **phase_kwargs)
            ax[1].set(title='Phase')
        return fig, ax
