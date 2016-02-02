from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from shampoo.reconstruction import Hologram
import matplotlib.pyplot as plt
hologram_path = 'data/Hologram.tif'
propagation_distance = 0.03685  # m

h = Hologram.from_tif(hologram_path)
waveform = h.reconstruct(propagation_distance,
                         plot_aberration_correction=False)

fig, ax = waveform.plot(all=True)
fig.suptitle("d = {0}".format(propagation_distance))
plt.show()
