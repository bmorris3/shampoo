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

def apodize(arr):
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
    n = np.shape(arr)[0]
    y, x = np.mgrid[0:n, 0:n]
    xmask = np.cos((x-n/2.)*np.pi/n)**0.25
    ymask = np.cos((y-n/2.)*np.pi/n)**0.25
    return arr*xmask*ymask

def generate_real_image_mask(n, center_x=285, center_y=200, radius=250):
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
    y, x = np.mgrid[0:n, 0:n]
    mask = np.zeros((n, n))
    mask[(x-center_x)**2 + (y-center_y)**2 < radius**2] = 1.0
    return mask

def fourier_trans_of_impulse_resp_func(propagation_distance, n,
                                       wavelength=405e-9, dx=3.45e-6, dy=3.45e-6):
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
    y, x = np.mgrid[0:n, 0:n] - n/2
    A = wavelength**2*(x+n**2*dx**2/(2.0*propagation_distance*wavelength))**2/(n**2*dx**2)
    B = wavelength**2*(y+n**2*dy**2/(2.0*propagation_distance*wavelength))**2/(n**2*dy**2)
    G = np.exp(-1j*2.0*np.pi*propagation_distance/wavelength*np.sqrt(1.0-A-B))
    return G

def calculate_reference_wave(psi, wavelength=405e-9, background_rows=[870, 140],
                             background_columns=[140, 870],
                             detector_edge_margin=100, plots=False):
    """
    Calculate the reference wave.

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
    detector_edge_margin : int
        Margin from edges of detector to avoid
    plots : bool
        Display plots after calculation if `True`

    Returns
    -------
    R : ndarray
        Reference wave array
    """
    n = np.shape(psi)[0]
    order = 3
    OPD = wavelength / (2*np.pi)
    y, x = np.mgrid[0:n, 0:n] - n/2
    pixvec = x[0, detector_edge_margin:-detector_edge_margin]
    img2 = np.fft.ifft2(psi)
    img2 = shift_peak(img2, [n/2, n/2])
    pimg2 = np.arctan(np.imag(img2)/np.real(img2))

    px = np.unwrap(pimg2[background_rows,
                   detector_edge_margin:-detector_edge_margin]*2.0)/2*OPD
    py = np.unwrap(pimg2[detector_edge_margin:-detector_edge_margin,
                   background_columns].T*2.0)/2*OPD
    px = np.mean(px, axis=0)
    py = np.mean(py, axis=0)

    pxpoly = np.polyfit(pixvec, px, order)
    pypoly = np.polyfit(pixvec, py, order)

    R = np.exp(-1j/OPD*(np.polyval(pxpoly, x) + np.polyval(pypoly, y)))
    if plots:
        fig, ax = plt.subplots(1,figsize=(15,8))
        ax.plot(pixvec, px/(wavelength/(2*np.pi)))
        ax.plot(pixvec, np.polyval(pxpoly, pixvec)/OPD, 'r--')

        plt.figure()
        plt.imshow(np.arctan(np.imag(R)/np.real(R)))#, origin='lower')
        plt.title('Fitted phase mask')
        plt.show()
    return R

def find_fourier_peak_centroid(fourier_arr, margin=50):
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
    n = np.shape(fourier_arr)[0]
    abs_fourier_arr = filters.gaussian_filter(np.abs(fourier_arr)[margin:n/2, margin:n/2], 10)
    spectrum_centroid = np.array(np.unravel_index(abs_fourier_arr.T.argmax(),
                                 abs_fourier_arr.shape)) + margin
    return spectrum_centroid

def reconstruct_wavefield(hologram, propagation_distance, wavelength=405e-9,
                         background_rows=[870, 140],
                         background_columns=[140, 870],
                         dx=3.45e-6, dy=3.45e-6,
                         detector_edge_margin=100, reference_wave=None):
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
    detector_edge_margin : int
        Margin from edges of detector to avoid

    Returns
    -------
    reconstructed_wavefield : ndarray (complex)
        Reconstructed wavefield from hologram

    reference_wave : ndarray
        Reference wave used to reconstruct wavefield
    """
    # Read input image
    n = np.shape(hologram)[0]
    apodized_hologram = apodize(hologram)

    # Isolate the real image in Fourier space, find spectral peak
    F_hologram = np.fft.fft2(apodized_hologram)
    spectrum_centroid = find_fourier_peak_centroid(F_hologram)

    # Create mask based on coords of spectral peak:
    mask = generate_real_image_mask(n, center_x=spectrum_centroid[0],
                                    center_y=spectrum_centroid[1])
    # Calculate Fourier transform of impulse response function
    G = fourier_trans_of_impulse_resp_func(propagation_distance, n,
                                           dx=dx, dy=dy)
    if reference_wave is None:
        # Center the spectral peak
        shifted_F_hologram = shift_peak(F_hologram*mask,
                                        [n/2-spectrum_centroid[1],
                                         n/2-spectrum_centroid[0]])

        # Apodize the result
        psi = apodize(shifted_F_hologram*G)
        # Calculate reference wave
        reference_wave = calculate_reference_wave(psi, wavelength=wavelength,
                                                  background_rows=background_rows,
                                                  background_columns=background_columns,
                                                  detector_edge_margin=detector_edge_margin,
                                                  plots=False)

    # Reconstruct the image
    psi = G*shift_peak(np.fft.fft2(apodized_hologram*reference_wave)*mask,
                       [n/2 - spectrum_centroid[1], n/2 - spectrum_centroid[0]])
    reconstructed_wavefield = shift_peak(np.fft.ifft2(psi), [n/2, n/2])
    return reconstructed_wavefield, reference_wave

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
    return tuple([i if not isinstance(i, list) else tuple(i)
                  for i in input_iterable])

class Hologram(object):
    """
    Container for holograms and their reference waves
    """
    def __init__(self, hologram_path):
        self.hologram_path = hologram_path
        self.hologram = self._load_hologram()
        self.reference_wave = None
        self.reconstructions = dict()

    def _load_hologram(self):
        """
        Load a hologram from path ``self.hologram_path`` using PIL and numpy.
        """
        return np.array(Image.open(self.hologram_path))

    def reconstruct(self, propagation_distance, wavelength=405e-9,
                    background_rows=[870, 140], background_columns=[140, 870],
                    dx=3.45e-6, dy=3.45e-6, detector_edge_margin=100):

        cache_key = make_items_hashable((propagation_distance, wavelength,
                                         background_rows,  background_columns,
                                         dx, dy, detector_edge_margin))
        if cache_key in self.reconstructions:
            reconstructed_wavefield = self.reconstructions[cache_key]

        else:
            reconstructed_wavefield, reference_wave = reconstruct_wavefield(self.hologram,
                                                                           propagation_distance,
                                                                           reference_wave=self.reference_wave)
            if self.reference_wave is None:
                self.reference_wave = reference_wave
            self.reconstructions[cache_key] = reconstructed_wavefield

        return ReconstructedWavefield(reconstructed_wavefield)

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
                          origin='lower')
                plt.show()
            elif intensity and not phase:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(self.phase_image[::-1,::-1], cmap=cmap,
                          origin='lower')
                plt.show()
        else:
            fig, ax = plt.subplots(1, 2, figsize=(18,8))
            ax[0].imshow(self.intensity_image[::-1,::-1], cmap=cmap,
                         origin='lower')
            ax[1].imshow(self.phase_image[::-1,::-1], cmap=cmap,
                         origin='lower')
            plt.show()
