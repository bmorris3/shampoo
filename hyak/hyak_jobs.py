from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import os
sys.path.insert(0, '/usr/lusers/bmmorris/git/shampoo/')

import numpy as np
from shampoo import Hologram, cluster_focus_peaks, locate_specimens
import datetime

print('Beginning task: ', sys.argv, datetime.datetime.utcnow())
hologram_path = sys.argv[1]
hologram_index = os.path.basename(hologram_path).split('_holo.tif')[0]
output_dir = sys.argv[2]
positions_path = os.path.join(output_dir, hologram_index+'_focus.txt')
coords_path = os.path.join(output_dir, hologram_index+'_coords.txt')

# Check that this hologram hasn't been done yet:
if not os.path.exists(coords_path):
    n_z_slices = 150
    distances = np.linspace(0.09, 0.14, n_z_slices)

    h = Hologram.from_tif(hologram_path, crop_fraction=2**-1)
    wave_cube = np.zeros((n_z_slices, h.n, h.n), dtype=np.complex128)
    positions = []
    for i, d in enumerate(distances):
        wave = h.reconstruct(d)
        wave_cube[i, ...] = wave.reconstructed_wave
        detected_positions = h.detect_specimens(wave, d)
        if detected_positions is not None:
            positions.append(detected_positions)

    positions = np.vstack(positions)

    # Compress along z axis for clustering
    positions_for_clustering = positions.copy()
    positions_for_clustering[:, 2] /= 10
    labels = cluster_focus_peaks(positions_for_clustering)

    coords, significance = locate_specimens(wave_cube, positions, labels,
                                            distances)

    # Save outputs
    coords_and_sig = np.column_stack([coords, significance])
    np.savetxt(coords_path, coords_and_sig)
