from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .reconstruction import ReconstructedWave

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

__all__ = ['cluster_focus_peaks', 'find_focus_plane']


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


def find_focus_plane(roi_cube, focus_on='amplitude', plot=False):
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

    Returns
    -------
    focus_index : int
        Index of the z-plane that is in focus
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
    focus_index = extremum(integral_abs_wave)

    if plot:
        plt.figure()
        plt.plot(range(roi_cube.shape[0]), integral_abs_wave)
        plt.axvline(focus_index, ls='--')

    return focus_index