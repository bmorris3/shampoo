from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from skimage.draw import circle_perimeter
import numpy as np
from numpy.testing import assert_allclose

from ..circlet import CircletTransform


def test_circlet():
    img = np.zeros((100, 100), dtype=float)

    cy, cx = 25, 35
    radius = 15
    rr1, cc1 = circle_perimeter(cy, cx, radius)
    img[rr1, cc1] = 1
    img += 0.5*np.random.rand(100, 100)

    c = CircletTransform(img)
    power = np.zeros_like(img)
    for test_radius in np.arange(radius-3, radius+3, 1):
        power += np.abs(c.coefficients(test_radius))

    power_centroid = np.unravel_index(np.argmax(power), power.shape)

    assert_allclose(power_centroid, [cy, cx], atol=0.1)
