from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .reconstruction import ReconstructedWave, unwrap_phase

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

__all__ = ['cluster_focus_peaks', 'find_focus_plane', 'locate_specimens']


def cluster_focus_peaks(xyz, eps=5, min_samples=3):
    """
    Use DBSCAN to identify single particles through multiple focus planes.

    Parameters
    ----------
    xyz : `~numpy.ndarray`
        Matrix of (x, y, z) positions for each peak detected
    eps : float
        Passed to the ``eps`` argument of `~sklearn.cluster.DBSCAN`
    min_samples : int
        Passed to the ``min_samples`` argument of `~sklearn.cluster.DBSCAN`

    Returns
    -------
    labels : `~numpy.ndarray`
        List of cluster labels for each peak. Labels of `-1` signify noise
        points.
    """
    positions = xyz.copy()

    # Compress distances in the z-axis
    positions[:, 2] /= 10

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    labels = db.labels_
    return labels


def find_focus_plane(roi_cube, median_cube, focus_on='phase', plot=False):
    """
    Find focus plane in a cube of reconstructed waves at different propagation
    distances.

    Uses the autofocus method of Dubois et al. 2006 who showed that the
    integral over the image plane of the amplitude of the reconstructed wave
    is minimum at the focal plane for a pure amplitude object [1]_. This will
    only work for small cubes centered on the specimen.

    .. [1] https://www.osapublishing.org/oe/abstract.cfm?uri=oe-14-13-5895

    Parameters
    ----------
    roi_cube : `~numpy.ndarray`
        Reconstructed waves at ``N`` propagation distances with ``M`` by ``M``
        pixels, with a shape of ``(N, M, M)``
    focus_on : {"amplitude", "phase"} (optional)
        Focus on the phase or amplitude?
    plot : bool (optional)
        Make plots of the integral of the amplitude of the reconstructed wave
        as a function of distance. Default is False.

    Returns
    -------
    focus_index : int
        Index of the z-plane that is in focus
    significance : float
        Significance of a detection of a particle in the ROI cube, measured by
        the significance of the negative peak in the derivative of the sum of
        the unwrapped phase in the image plane, with respect to propagation
        distance. For example, if ``significance < 3`` the detection of a
        specimen may be legitimate.
    """
    if focus_on == 'amplitude':
        extremum = np.argmin
    elif focus_on == 'phase':
        extremum = np.argmax
    else:
        raise ValueError('The `focus_on` kwarg must be either "phase" or '
                         '"amplitude".')

    # Following Equation 9, 10 of Dubois et al. 2006:
    integral_abs_wave = np.sum(np.abs(roi_cube), axis=(1, 2))
    integral_phase_wave = np.sum(np.arctan(roi_cube.imag / roi_cube.real), axis=(1, 2))
    #focus_index = extremum(integral_abs_wave)
    focus_min = np.argmin(integral_abs_wave)
    focus_max = np.argmax(integral_abs_wave)
    focus_index = focus_min #0.5*(focus_min + focus_max)
    # Do a similar integral on the unwrapped phase. The phase changes
    # most rapidly on a source near focus, so the derivative wrt propagation
    # distance of the phase integrated in space has a *minimum* near focus
    # unwrapped_phase_cube = [unwrap_phase(roi_cube[i, ...])
    #                         for i in range(roi_cube.shape[0])]
    # integral_phase_wave = np.array([np.ptp(upc) for upc in unwrapped_phase_cube])
    # intensity = np.abs(roi_cube)
    #integral_phase_wave = np.sum(intensity - np.median(intensity, axis=0), axis=(1, 2))

    from astropy.convolution import convolve_fft, Gaussian2DKernel

    # conv = [convolve_fft(i, Gaussian2DKernel(4),
    #                      complex_dtype=np.complex128).sum()
    #         for i in intensity]
    # integral_phase_wave = np.array(conv)
    #plt.plot(conv)
    # fig, ax = plt.subplots(1, len(conv))
    # for i, c in enumerate(intensity):
    #     ax[i].imshow(c)
    #     ax[i].set_xticks([])
    #     ax[i].set_yticks([])
    #     if i == focus_index:
    #         ax[i].set_title('X')
    #plt.show()

    #d_int_phase = np.diff(integral_phase_wave)

    # Measure significance of detected focus by taking the median normalized,
    # standard deviation normalized derivative of phase with respect to
    # propagation distance. This quantity becomes significantly negative for
    # real specimens. For example, significance < -3 may yield real particles.
    significance = 0.1#np.min(d_int_phase / np.median(d_int_phase) /
                       #   d_int_phase.std())
    if plot:
        plt.figure()
        plt.plot(range(roi_cube.shape[0]),
                 (integral_abs_wave - integral_abs_wave.mean())/integral_abs_wave.std(),
                 label='intensity')

        # for mc in median_cube:
        #     plt.plot(range(roi_cube.shape[0]),
        #              (mc - mc.mean())/mc.std(),
        #              label='intensity')

        # plt.plot(range(roi_cube.shape[0]-1),
        #          (d_int_phase - d_int_phase.mean())/d_int_phase.std(),
        #          label='d(phase)/d(prop dist)')

        # plt.plot(range(roi_cube.shape[0]-1),
        #          (np.abs(d_int_phase) - d_int_phase.mean())/d_int_phase.std(),
        #          label='|d(phase)/d(prop dist)|')

        plt.plot(range(roi_cube.shape[0]),
                 (integral_phase_wave - integral_phase_wave.mean()) /
                 integral_phase_wave.std(),
                 label='phase')

        plt.xlabel('Propagation distance index')
        plt.title('Significance: {0:.4f}'.format(significance))
        plt.axvline(focus_index, ls='--', lw=2)
        plt.axvline(focus_min, ls='--')
        plt.axvline(focus_max, ls='--')

        plt.legend()
    return focus_index, significance


def _correct_limits(minimum, maximum, axis_range, edge):
    if minimum < axis_range:
        minimum = axis_range
    if maximum > edge - axis_range:
        maximum = edge - axis_range
    return minimum, maximum, axis_range


def locate_specimens(wave_cube, positions, labels, distances, plots=False):
    """
    Identify the (x, y, z) coordinates of a specimen.

    Parameters
    ----------
    wave_cube : `~numpy.ndarray`
        Cube of reconstructed waves
    positions : `~numpy.ndarray`
        (x,y,z) positions of objects detected by the blob finder
    labels : `~numpy.ndarray`
        Clustering labels for each (x,y,z) coordinate, identifying groups
        of positions, i.e., single particles detected at multiple z-planes
    distances : `~numpy.ndarray`
        Propagation distances, same length as the first axis of ``complex_cube``

    Returns
    -------
    specimen_coordinates : `~numpy.ndarray`
        (x, y, z) coordinates of each detected specimen
    specimen_coordinates : `~numpy.ndarray`
        Significance of each specimen detection. See docs of
        `~shampoo.focus.find_focus_plane` for hints on how to interpret
        the significance quantity.
    """
    specimen_coordinates = []
    specimen_significance = []
    for l in set(labels):
        n_points = np.count_nonzero(labels == l)
        if l != -1 and n_points > 3:
            xmedian = np.median(positions[labels == l, 0])
            ymedian = np.median(positions[labels == l, 1])
            xmin, ymin, zmin_d = np.min(positions[labels == l, :], axis=0)
            xmax, ymax, zmax_d = np.max(positions[labels == l, :], axis=0)
            zmin = np.argmin(np.abs(zmin_d - distances))
            zmax = np.argmin(np.abs(zmax_d - distances))

            x_range = y_range = 2
            z_range = zmax - zmin

            xmin, xmax, x_range = _correct_limits(xmin, xmax, x_range,
                                                  wave_cube.shape[1])
            ymin, ymay, y_range = _correct_limits(ymin, ymax, y_range,
                                                  wave_cube.shape[2])
            zmin, zmaz, z_range = _correct_limits(zmin, zmax, z_range,
                                                  wave_cube.shape[0])

            # Make a reconstructed wave cube centered on the region of interest
            roi_cube = wave_cube[zmin - z_range:zmax + z_range,
                                 xmin - x_range:xmax + x_range,
                                 ymin - y_range:ymax + y_range]

            # Using this cropped cube centered on the ROI, find the best focus

            focus_ind_minus_margin, significance = find_focus_plane(roi_cube,
                                                                    plot=plots)
            focus_ind = focus_ind_minus_margin + zmin - z_range

            specimen_coordinates.append([xmedian, ymedian,
                                         distances[focus_ind]])
            specimen_significance.append(significance)

            if plots:
                focused_wave = ReconstructedWave(wave_cube[focus_ind, ...])

                fig, ax = focused_wave.plot(phase=True)
                thetas = np.linspace(0, 2*np.pi, 30)
                r = 20
                ax.plot(r*np.cos(thetas) + ymedian,
                        r*np.sin(thetas) + xmedian, lw=3, color='r')
                plt.show()

    return np.array(specimen_coordinates), np.array(specimen_significance)
