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
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.ndimage import filters

from skimage.restoration import unwrap_phase
from skimage.filters.rank import noise_filter
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
    Load a hologram from path ``self.hologram_path`` using PIL and numpy.
    """
    return np.array(Image.open(hologram_path), dtype=np.float64)


class Hologram(object):
    """
    Container for holograms and their reference wavefields
    """
    def __init__(self, hologram, wavelength=405e-9,
                 rebin_factor=1, dx=3.45e-6, dy=3.45e-6,
                 background_interval=3):
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
        self.hologram = rebin_image(hologram, self.rebin_factor)
        self.n = self.hologram.shape[0]
        self.wavelength = wavelength
        self.wavenumber = 2*np.pi/self.wavelength
        self.reconstructions = dict()
        self.dx = dx*rebin_factor
        self.dy = dy*rebin_factor
        self.mgrid = np.mgrid[0:self.n, 0:self.n]
        self.random_seed = RANDOM_SEED

    @classmethod
    def from_tif(cls, hologram_path, **kwargs):
        """
        Open a raw hologram from a tif file.

        Parameters
        ----------
        hologram_path : string
            Path to input hologram to reconstruct

        Notes
        -----
        All keyword arguments get passed on to the `~shampoo.Hologram`
        constructor.
        """
        hologram = _load_hologram(hologram_path)
        return cls(hologram, **kwargs)

    def reconstruct(self, propagation_distance,
                    plot_aberration_correction=False,
                    plot_fourier_peak=False,
                    cache=False, reference_wave=None):
        """
        Wrapper around reconstruct_wavefield.

        Parameters
        ----------
        reference_wave : `~numpy.ndarray` or `None` (optional)
            If one has been calculated, you can give a reference wave
            to use.
        """
        self.reference_wave = reference_wave

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
        hologram : ndarray
            Raw input hologram
        propagation_distance : float
            Propagation distance [m]
        R : ndarray or `None` (optional)
            If `None`, compute the reference wave, else use array as reference wave
            array for reconstruction

        Returns
        -------
        reconstructed_wavefield : ndarray (complex)
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

        if self.reference_wave is None:
            # Center the spectral peak
            shifted_F_hologram = shift_peak(F_hologram*mask,
                                            [self.n/2-spectrum_centroid[1],
                                             self.n/2-spectrum_centroid[0]])

            # Apodize the result
            psi = self.apodize(shifted_F_hologram*G)
            #psi = shifted_F_hologram*G

            # Calculate reference wave
            self.reference_wave = self.get_digital_phase_mask(psi,
                                                              plots=plot_aberration_correction)

        # Reconstruct the image
        psi = G*shift_peak(fft2(apodized_hologram*self.reference_wave)*mask,
                           [self.n/2 - spectrum_centroid[1],
                            self.n/2 - spectrum_centroid[0]])

        reconstructed_wavefield = shift_peak(ifft2(psi),
                                             [self.n/2, self.n/2])
        return reconstructed_wavefield

    def get_digital_phase_mask(self, psi, plots=False, aberr_corr_order=3,
                               n_aberr_corr_iter=2):
        """
        Calculate the digital phase mask (i.e. reference wave), as in Colomb et
        al. 2006, Eqn. 26 [1]_.

        .. [1] http://www.ncbi.nlm.nih.gov/pubmed/16512526

        Parameters
        ----------
        psi : ndarray
            The product of the Fourier transform of the hologram and the Fourier
            transform of impulse response function
        plots : bool
            Display plots after calculation if `True`
        aberr_corr_order : int
            Polynomial order of the aberration corrections
        n_aberr_corr_iter : int
            Number of aberration correction iterations

        Returns
        -------
        R : ndarray
            Reference wave array
        """
        order = aberr_corr_order #3#2

        y, x = self.mgrid - self.n/2

        inverse_psi = shift_peak(ifft2(psi), [self.n/2, self.n/2])

        phase_image = np.arctan(np.imag(inverse_psi) / np.real(inverse_psi))
        unwrapped_phase_image = unwrap_phase(2*phase_image,
                                             seed=self.random_seed)/2/self.wavenumber
        unwrapped_phase_image = gaussian_filter(unwrapped_phase_image, 3)

        grad_x_2 = np.sum(np.abs(np.diff(unwrapped_phase_image, axis=0)), axis=1)
        grad_y_2 = np.sum(np.abs(np.diff(unwrapped_phase_image, axis=1)), axis=1)

        n_rows_or_cols = 100
        least_noisy_x_indices = grad_x_2.argsort()[:n_rows_or_cols]
        least_noisy_y_indices = grad_y_2.argsort()[:n_rows_or_cols]

        # Compute the smoothed mean phase in the background along each axis
        phase_x_bg = np.mean(unwrapped_phase_image[:, least_noisy_x_indices], axis=1)
        phase_y_bg = np.mean(unwrapped_phase_image[least_noisy_y_indices, :].T, axis=1)
        phase_x_bg_smooth = gaussian_filter1d(phase_x_bg, 50)
        phase_y_bg_smooth = gaussian_filter1d(phase_y_bg, 50)

        pixel_indices = x[0, :]
        phase_x_polynomials = np.polyfit(pixel_indices, phase_x_bg_smooth,
                                         order)
        phase_y_polynomials = np.polyfit(pixel_indices, phase_y_bg_smooth,
                                         order)

        # Iterate to fit the median of typical rows/columns multiple times
        background_x_polynomials = phase_x_polynomials.copy()
        background_y_polynomials = phase_y_polynomials.copy()
        for n in range(n_aberr_corr_iter):
            background_x_polynomials += np.polyfit(pixel_indices,
                                                   phase_x_bg_smooth -
                                                   np.polyval(background_x_polynomials,
                                                              pixel_indices), order)
            background_y_polynomials += np.polyfit(pixel_indices,
                                                   phase_y_bg_smooth -
                                                   np.polyval(background_y_polynomials,
                                                              pixel_indices), order)

        digital_phase_mask = np.exp(-1j*self.wavenumber *
                                    (np.polyval(background_x_polynomials, x) +
                                     np.polyval(background_y_polynomials, y)))

        if plots:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax[0, 0].plot(grad_x_2)
            ax[1, 0].imshow(unwrapped_phase_image,
                            cmap=plt.cm.binary_r, origin='lower')
            ax[1, 1].plot(grad_y_2, range(len(grad_y_2)))
            ax[1, 1].set(ylim=(0, len(grad_y_2)))
            ax[0, 0].set(xlim=(0, len(grad_x_2)))

            for x_i, y_i in zip(least_noisy_x_indices, least_noisy_y_indices):
                ax[0, 0].axvline(x_i, color='m', alpha=0.4, ls=':')
                ax[1, 1].axhline(y_i, color='m', alpha=0.4, ls=':')

                ax[1, 0].axvline(x_i, color='m', alpha=0.4, ls=':')
                ax[1, 0].axhline(y_i, color='m', alpha=0.4, ls=':')
            plt.show()

        return digital_phase_mask

    def apodize(self, arr):
        """
        Force the magnitude at the boundaries go to zero

        Parameters
        ----------
        arr : ndarray
            Array to apodize

        Returns
        -------
        apodized_arr : ndarray
            Apodized array
        """
        y, x = self.mgrid
        arr *= (np.sqrt(np.cos((x-self.n/2.)*np.pi/self.n)) *
                np.sqrt(np.cos((y-self.n/2.)*np.pi/self.n)))
        return arr

    def fourier_trans_of_impulse_resp_func(self, propagation_distance):
        """
        Calculate the Fourier transform of impulse response function, often written
        as ``G``.

        For reference, see Eqn 3.22 of Schnars & Juptner (2002) Meas. Sci. Technol.
        13 R85-R101 [1]_.

        .. [1] http://x-ray.ucsd.edu/mediawiki/images/d/df/Digital_recording_numerical_reconstruction.pdf

        Parameters
        ----------
        n : int
            Number of pixels on each side of the image
        wavelength : float
            Wavelength of beam [m]
        dx : float
            Pixel size [m]
        dy : float
            Pixel size [m]

        Returns
        -------
        G : ndarray
            Fourier transform of impulse response function
        """
        y, x = self.mgrid - self.n/2
        a = (self.wavelength**2 * (x + self.n**2 * self.dx**2 /
             (2.0 * propagation_distance * self.wavelength))**2 /
             (self.n**2 * self.dx**2))
        b = (self.wavelength**2 * (y + self.n**2 * self.dy**2 /
             (2.0 * propagation_distance * self.wavelength))**2 /
             (self.n**2*self.dy**2))
        g = np.exp(-1j * self.wavenumber* propagation_distance *
                   np.sqrt(1.0 - a - b))
        return g

    def generate_real_image_mask(self, center_x, center_y, radius):
        """
        Calculate the Fourier-space mask to isolate the real image
    
        Parameters
        ----------
        n : int
            Number of pixels on each side of the image
        center_x : int
            ``x`` centroid [pixels] of real image in Fourier space
        center_y : int
            ``y`` centroid [pixels] of real image in Fourier space
        radius : float
            Radial width of mask [pixels] to apply to the real image in Fourier
            space
    
        Returns
        -------
        mask : ndarray
            Binary array, with values corresponding to `True` within the mask
            and `False` elsewhere.
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
        fourier_arr : ndarray
        Fourier transformed hologram
        margin : int
        Avoid the nearest ``margin`` pixels from the edge of detector when
        looking for the peak

        Returns
        -------
        pixel
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
    incarnations.
    """
    def __init__(self, reconstructed_wavefield):
        self.reconstructed_wavefield = reconstructed_wavefield
        self._intensity_image = None
        self._phase_image = None
        self.random_seed = RANDOM_SEED

    @property
    def intensity(self):
        if self._intensity_image is None:
            self._intensity_image = np.abs(self.reconstructed_wavefield)
        return self._intensity_image

    @property
    def phase(self):
        if self._phase_image is None:
            self._phase_image = unwrap_phase(2*np.arctan(np.imag(self.reconstructed_wavefield)/
                                             np.real(self.reconstructed_wavefield)),
                                             seed=self.random_seed)

        return self._phase_image

    def plot(self, phase=False, intensity=False, all=None, cmap=plt.cm.binary_r):

        phase_kwargs = {}

        if all is None:
            if phase and not intensity:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(self.phase[::-1,::-1], cmap=cmap,
                          origin='lower', interpolation='nearest',
                          **phase_kwargs)
            elif intensity and not phase:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(self.intensity[::-1,::-1], cmap=cmap,
                          origin='lower', interpolation='nearest')
        else:
            fig, ax = plt.subplots(1, 2, figsize=(18,8), sharex=True, sharey=True)
            ax[0].imshow(self.intensity[::-1,::-1], cmap=cmap,
                         origin='lower', interpolation='nearest')
            ax[0].set(title='Intensity')
            ax[1].imshow(self.phase[::-1,::-1], cmap=cmap,
                         origin='lower', interpolation='nearest',
                          **phase_kwargs)
            ax[1].set(title='Phase')
        return fig, ax
