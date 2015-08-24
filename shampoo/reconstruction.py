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
import matplotlib.pyplot as plt
from scipy.ndimage import filters

def rebin_image(a, binning_factor):
    # Courtesy of J.F. Sebastian: http://stackoverflow.com/a/8090605
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

class Hologram(object):
    """
    Container for holograms and their reference wavefields
    """
    def __init__(self, hologram_path, wavelength=405e-9,
                 rebin_factor=1, dx=3.45e-6, dy=3.45e-6,
                 detector_edge_margin=0.15, background_interval=3):
        """
        Parameters
        hologram_path : string
            Path to input hologram to reconstruct
        wavelength : float [meters]
            Wavelength of laser
        rebin_factor : int
            Rebin the image by factor ``rebin_factor``. Must be an even integer.
        dx : float [meters]
            Pixel width in x-direction (unbinned)
        dy: float [meters]
            Pixel width in y-direction (unbinned)
        detector_edge_margin : float
            Fraction of total detector width to ignore on the edges
        """
        self.hologram_path = hologram_path
        self.rebin_factor = rebin_factor
        self.hologram = rebin_image(self._load_hologram(), self.rebin_factor)
        self.n = self.hologram.shape[0]
        self.reference_wave = None
        self.wavelength = wavelength
        self.wavenumber = 2*np.pi/self.wavelength
        self.reconstructions = dict()
        self.dx = dx*rebin_factor
        self.dy = dy*rebin_factor
        self.edge_margin = int(self.n/rebin_factor*detector_edge_margin)
        self.background_columns = np.arange(0, self.n, background_interval)
        self.background_rows = np.arange(0, self.n, background_interval)
        self.mgrid = np.mgrid[0:self.n, 0:self.n]

    def _load_hologram(self):
        """
        Load a hologram from path ``self.hologram_path`` using PIL and numpy.
        """
        return np.array(Image.open(self.hologram_path))

    def reconstruct(self, propagation_distance,
                    plot_aberration_correction=False):

        cache_key = make_items_hashable((propagation_distance, self.wavelength,
                                         self.background_rows,
                                         self.background_columns, self.dx,
                                         self.dy, self.edge_margin))
        if cache_key not in self.reconstructions:
            reconstructed_wavefield, reference_wave = self.reconstruct_wavefield(propagation_distance,
                                                                                 plot_aberration_correction=plot_aberration_correction)
            if self.reference_wave is None:
                self.reference_wave = reference_wave
            self.reconstructions[cache_key] = ReconstructedWavefield(reconstructed_wavefield)

        return self.reconstructions[cache_key]

    def reconstruct_wavefield(self, propagation_distance,
                              plot_aberration_correction=False):
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
        wavelength : float
            Wavelength of beam [m]
        background_rows : list
            Rows in the reconstructed image where there is no specimen
        background_columns : list
            Columns in the reconstructed image where there is no specimen
        edge_margin : int
            Margin from edges of detector to avoid

        Returns
        -------
        reconstructed_wavefield : ndarray (complex)
            Reconstructed wavefield from hologram

        reference_wave : ndarray
            Reference wave used to reconstruct wavefield
        """
        # Read input image
        apodized_hologram = self.apodize(self.hologram)

        # Isolate the real image in Fourier space, find spectral peak
        F_hologram = np.fft.fft2(apodized_hologram)
        spectrum_centroid = self.find_fourier_peak_centroid(F_hologram)

        # Create mask based on coords of spectral peak:
        mask_radius = 150/self.rebin_factor
        mask = self.generate_real_image_mask(spectrum_centroid[0],
                                             spectrum_centroid[1],
                                             mask_radius)
        # Calculate Fourier transform of impulse response function
        G = self.fourier_trans_of_impulse_resp_func(propagation_distance)

        if self.reference_wave is None:
            # Center the spectral peak
            shifted_F_hologram = shift_peak(F_hologram*mask,
                                            [self.n/2-spectrum_centroid[1],
                                             self.n/2-spectrum_centroid[0]])

            # Apodize the result
            #psi = self.apodize(shifted_F_hologram*G)
            psi = shifted_F_hologram*G

            # Calculate reference wave
            reference_wave = self.calculate_digital_phase_mask(psi,
                                                               plots=plot_aberration_correction)

        # Reconstruct the image
        psi = G*shift_peak(np.fft.fft2(apodized_hologram*reference_wave)*mask,
                           [self.n/2 - spectrum_centroid[1],
                            self.n/2 - spectrum_centroid[0]])
        reconstructed_wavefield = shift_peak(np.fft.ifft2(psi),
                                             [self.n/2, self.n/2])
        return reconstructed_wavefield, reference_wave

    def calculate_digital_phase_mask(self, psi, plots=False, aberr_corr_order=3,
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
        wavelength : float
            Wavelength of beam [m]
        background_rows : list
            Rows in the reconstructed image where there is no specimen
        background_columns : list
            Columns in the reconstructed image where there is no specimen
        edge_margin : int
            Margin from edges of detector to avoid
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
        order = aberr_corr_order#3#2
        y, x = self.mgrid - self.n/2
        pixel_indices = x[0, self.edge_margin:-self.edge_margin]
        inverse_psi = shift_peak(np.fft.ifft2(psi), [self.n/2, self.n/2])
        phase_image = np.arctan(np.imag(inverse_psi) / np.real(inverse_psi))

        phase_x = (np.unwrap(2*phase_image[self.background_rows,
                   self.edge_margin:-self.edge_margin])/2 /
                   self.wavenumber)
        phase_y = (np.unwrap(2*phase_image[self.edge_margin:-self.edge_margin,
                   self.background_columns].T)/2/self.wavenumber)

        if plots:
            phase_x_before_median = phase_x.copy()
            phase_y_before_median = phase_y.copy()

        # Since you can't know which rows/cols are background rows/cols a
        # priori, find the phase profiles along each row, fit polynomials to
        # each, measure the rms error to the fit. Do a second fit excluding
        # high error rows/cols and taking the median of the good ones.

        # initial median fit to rows/cols
        pxpoly_init = np.polyfit(pixel_indices, np.median(phase_x, axis=0),
                                 order)
        pypoly_init = np.polyfit(pixel_indices, np.median(phase_y, axis=0),
                                 order)

        # subtract each row by initial fit, take std to measure error
        rms_x = np.std(np.polyval(pxpoly_init, pixel_indices)[:,np.newaxis] -
                       phase_x.T, axis=0)
        rms_y = np.std(np.polyval(pypoly_init, pixel_indices)[:,np.newaxis] -
                       phase_y.T, axis=0)

        x_within_some_sigma = (np.abs(rms_x - np.median(rms_x)) <
                               0.5*np.std(rms_x))
        y_within_some_sigma = (np.abs(rms_y - np.median(rms_y)) <
                               0.5*np.std(rms_y))

        # First fit to the median of only the typical rows/columns:
        phase_x_background = np.median(phase_x[x_within_some_sigma], axis=0)
        phase_y_background = np.median(phase_y[y_within_some_sigma], axis=0)
        phase_x_polynomials = np.polyfit(pixel_indices, phase_x_background,
                                         order)
        phase_y_polynomials = np.polyfit(pixel_indices, phase_y_background,
                                         order)

        # Iterate to fit the median of typical rows/columns multiple times
        background_x_polynomials = phase_x_polynomials.copy()
        background_y_polynomials = phase_y_polynomials.copy()
        for n in range(n_aberr_corr_iter):
            background_x_polynomials += np.polyfit(pixel_indices,
                                        phase_x_background -
                                        np.polyval(background_x_polynomials,
                                                   pixel_indices),
                                        order)
            background_y_polynomials += np.polyfit(pixel_indices,
                                        phase_y_background -
                                        np.polyval(background_y_polynomials,
                                                   pixel_indices),
                                        order)

        digital_phase_mask = np.exp(-1j*self.wavenumber *
                                    (np.polyval(background_x_polynomials, x) +
                                     np.polyval(background_y_polynomials, y)))

        if plots:
            fig, ax = plt.subplots(1,3,figsize=(16,8))
            ax[0].imshow(np.unwrap(phase_image), origin='lower')
            [ax[0].axhline(background_row)
             for background_row in self.background_rows[x_within_some_sigma]]
            [ax[0].axvline(background_column)
             for background_column in self.background_columns[y_within_some_sigma]]

            ax[1].plot(pixel_indices,
                       phase_x_before_median[x_within_some_sigma].T *
                       self.wavenumber)
            ax[1].plot(pixel_indices,
                       np.polyval(phase_x_polynomials, pixel_indices) *
                       self.wavenumber, 'r', lw=4)
            ax[1].set_title('rows (x)')

            ax[2].plot(pixel_indices,
                       phase_y_before_median[y_within_some_sigma].T *
                       self.wavenumber)
            ax[2].plot(pixel_indices,
                       np.polyval(phase_y_polynomials, pixel_indices) *
                       self.wavenumber, 'r', lw=4)
            ax[2].set_title('cols (y)')
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
        suppression_exponent = 0.4
        arr *= (np.cos((x-self.n/2.)*np.pi/self.n)**suppression_exponent *
                np.cos((y-self.n/2.)*np.pi/self.n)**suppression_exponent)
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

    @property
    def intensity_image(self):
        if self._intensity_image is None:
            self._intensity_image = np.abs(self.reconstructed_wavefield)
        return self._intensity_image

    @property
    def phase_image(self):
        if self._phase_image is None:
            self._phase_image = np.arctan(np.imag(self.reconstructed_wavefield)/
                                          np.real(self.reconstructed_wavefield))
        return self._phase_image

    def plot(self, phase=False, intensity=False, all=None, cmap=plt.cm.binary_r):
        if all is None:
            if phase and not intensity:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(self.intensity_image[::-1,::-1], cmap=cmap,
                          origin='lower', interpolation='nearest')
                plt.show()
            elif intensity and not phase:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(self.phase_image[::-1,::-1], cmap=cmap,
                          origin='lower', interpolation='nearest')
                plt.show()
        else:
            fig, ax = plt.subplots(1, 2, figsize=(18,8), sharex=True, sharey=True)
            ax[0].imshow(self.intensity_image[::-1,::-1], cmap=cmap,
                         origin='lower', interpolation='nearest')
            ax[1].imshow(self.phase_image[::-1,::-1], cmap=cmap,
                         origin='lower', interpolation='nearest')
        return fig, ax
