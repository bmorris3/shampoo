from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from shampoo import Hologram

hologram_path = ('data/00162_holo.tif')
propagation_distance = 0.1202  # m

h = Hologram.from_tif(hologram_path, crop_fraction=2**-1)
waveform = h.reconstruct(propagation_distance)

import matplotlib.pyplot as plt
fig, ax = waveform.plot(phase=True)
fig.suptitle("d = {0}".format(propagation_distance))
plt.show()
