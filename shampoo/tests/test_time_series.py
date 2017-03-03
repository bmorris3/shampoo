# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os.path
from ..reconstruction import Hologram, ReconstructedWave, random_seed
import tempfile
from ..time_series import TimeSeries

np.random.seed(random_seed)

def _example_hologram(dim = 512):
    """ Generate example Hologram object """
    im = 1000*np.ones((dim, dim), dtype = np.float) + np.random.random(size = (dim, dim))
    return Hologram(im)

def test_time_series_metadata_defaults():
    name = os.path.join(tempfile.gettempdir(), 'test_time_series.hdf5')
    with TimeSeries(filename = name, mode = 'w') as time_series:

        # Check default values when newly-created object
        assert time_series.time_points == tuple()
        assert time_series.propagation_distances == tuple()
        assert time_series.wavelengths == tuple()

def test_time_series_metadata():
    """ """
    name = os.path.join(tempfile.gettempdir(), 'test_time_series.hdf5')
    with TimeSeries(filename = name, mode = 'w') as time_series:

        #Iterables should be sorted
        time_series.time_points = (1,2,4,3,5)
        assert time_series.time_points == (1,2,3,4,5)

        time_series.propagation_distances = (5,3,4,2,1)
        assert time_series.propagation_distances == (1,2,3,4,5)

        time_series.wavelengths = (4,3,2,5,1)
        assert time_series.wavelengths == (1,2,3,4,5)

def test_time_series_storing_hologram():
    """ Test storage of holograms """
    name = os.path.join(tempfile.gettempdir(), 'test_time_series.hdf5')
    hologram = _example_hologram()
    with TimeSeries(filename = name, mode = 'w') as time_series:
        time_series.add_hologram(hologram, time_point = 0)

        assert time_series.time_points == (0,)

        retrieved = time_series.hologram(0)
        assert np.allclose(hologram.hologram, retrieved.hologram)

def test_time_series_reconstruct():
    name = os.path.join(tempfile.gettempdir(), 'test_time_series.hdf5')
    hologram = _example_hologram()

    with TimeSeries(filename = name, mode = 'w') as time_series:
        time_series.add_hologram(hologram, time_point = 0)
        ts_reconw = time_series.reconstruct(time_point = 0, 
                                            wavelength = hologram.wavelength,
                                            propagation_distance = 0.03)
        assert isinstance(ts_reconw, ReconstructedWave) 
        
        # Retrieve reconstructed wave from archive
        archived_reconw = time_series.reconstructed_wave(time_point = 0, 
                                                         wavelength = hologram.wavelength)
        assert isinstance(archived_reconw, ReconstructedWave) 
                           