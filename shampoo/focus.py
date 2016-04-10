from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from sklearn.cluster import DBSCAN

__all__ = ['cluster_focus_peaks']


def cluster_focus_peaks(xyz, eps=5, min_samples=3):
    """
    Use DBSCAN to identify single particles through multiple focus planes.

    Parameters
    ----------
    xyz : `~numpy.ndarray`
        Matrix of (x, y, z) positions for each peak detected

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

