from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from sklearn.cluster import DBSCAN

__all__ = ['cluster_focus_peaks']


def cluster_focus_peaks(xyz):
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
    db = DBSCAN(eps=5, min_samples=2).fit(xyz)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    return labels

