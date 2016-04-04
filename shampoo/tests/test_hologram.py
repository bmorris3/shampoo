from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..reconstruction import Hologram, rebin_image, _find_peak_centroid

import numpy as np


RANDOM_SEED = 40
np.random.seed(RANDOM_SEED)


def _example_hologram(dim=2048):
    """
    Generate example hologram.

    Parameters
    ----------
    dim : int
        Dimensions of image. Default is 2048.
    """
    return 1000*np.ones((dim, dim)) + np.random.randn(dim, dim)


def test_load_hologram():
    holo = Hologram(_example_hologram())


def test_rebin_image():
    dim = 2048
    full_res = _example_hologram(dim=dim)
    assert (dim//2, dim//2) == rebin_image(full_res, 2).shape


def _gaussian2d(amplitude, width, centroid, dim):
    x, y = np.mgrid[0:dim, 0:dim]
    x_centroid, y_centroid = centroid
    return amplitude*np.exp(-0.5 * ((x - x_centroid)**2/width**2 +
                                    (y - y_centroid)**2/width**2))


def test_centroid():
    centroid = (65, 35)
    test_image = _gaussian2d(amplitude=10, width=5, centroid=centroid, dim=100)
    assert np.all(_find_peak_centroid(image=test_image) == centroid)
