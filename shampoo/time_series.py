# -*- coding: utf-8 -*-
"""
This module implements storage of holographic time-series
via an HDF5 file.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import Iterable
import h5py
import numpy as np
import os
from .reconstruction import Hologram, ReconstructedWave
from skimage.io import imread

class Metadata(object):
    """ 
    Metadata of a TimeSeries object as a descriptor.
    
    Note
    ----
    Iterable metadata is kept sorted.
    """
    def __init__(self, name, output):
        """
        Parameters
        ----------
        name : string
        output : callable
        """
        self.name = name
        self.output = output
    
    def __get__(self, instance, cls):
        return self.output(instance['/'].attrs[self.name])
    
    def __set__(self, instance, value):
        if isinstance(value, Iterable):
            value = tuple(sorted(value))
        instance['/'].attrs[self.name] = value
    
    def __delete__(self, instance):
        del instance['/'].attrs[self.name]

class TimeSeries(h5py.File):
    """
    Holographic time-series as an HDF5 archive.

    Attributes
    ----------
    time_points : tuple of floats
        Time-points in seconds.
    propagation_distances : tuple of floats
        Propagation distances in meters.
    wavelengths : tuple of floats
        Wavelengths in nm.
    
    Notes
    -----
    The underlying HDF5 file has the following structure:
    /t0000/hologram
    /t0000/wavelength0/intensity
    /t0000/wavelength0/phase
    /t0000/wavelength0/fourier_mask
    /t0000/wavelength1/intensity
    /t0000/wavelength1/phase
    /t0000/wavelength1/fourier_mask
    ...
    /t0001/hologram
    /t0001/wavelength0/intensity
    /t0001/wavelength0/phase
    /t0001/wavelength0/fourier_mask
    """

    # metadata is exposed as tuples because it is immutable
    time_points = Metadata('time_points', output = tuple)
    propagation_distances = Metadata('propagation_distances', output = tuple)
    wavelengths = Metadata('wavelengths', output = tuple)

    def __init__(self, filename, mode = 'r', **kwargs):
        """
        Parameters
        ----------
        filename : string
            Path to the HDF5 archive.
        mode : string, optional
            File-mode. Possible values in 
            {'r', 'r+', 'w+', 'a'}
        """
        super(TimeSeries, self).__init__(name = filename, mode = mode, **kwargs)

        # Check that metadata attributes exist
        # Otherwise, set to empty
        for attr in ('time_points', 'propagation_distances', 'wavelengths'):
            try:
                getattr(self, attr)
            except KeyError:
                setattr(self, attr, tuple())
    
    def hologram(self, time_point, **kwargs):
        """
        Return Hologram object from archive. Keyword arguments are
        passed to the Hologram constructor.
        
        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        
        Returns
        -------
        out : Hologram
        """
        data = np.array(self._time_group(time_point)['hologram'])
        return Hologram(data, **kwargs)
    
    def reconstructed_wave(self, time_point, wavelength):
        """
        Returns the ReconstructedWave object from archive. 

        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        wavelength : int
            Wavelength in nm.
        
        Returns
        -------
        out : ReconstructedWave object
        """
        gp = self._reconstruction_group(time_point, wavelength)
        return ReconstructedWave(np.array(gp['intensity']) + 1j*np.array(gp['phase']))
    
    def reconstruct(self, time_point, wavelength, propagation_distance, **kwargs):
        """
        Hologram reconstruction.
        
        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        wavelength : int
            Wavelength in nm.
        propagation_distance : float
            Propagation distance in meters.
        
        Returns
        -------
        out : ReconstructedWave object
        """
        if wavelength not in self.wavelengths:
            self.wavelengths = self.wavelengths + (wavelength,)
        
        # TODO: extend to multiple propagation distances
        hologram = self.hologram(time_point)
        recon_wave = hologram.reconstruct(propagation_distance, **kwargs)

        # TODO: store reconstruction parameters, e.g. fourier mask
        # TODO: overwrite dataset if already exists
        gp = self._reconstruction_group(time_point, wavelength)
        gp.create_dataset(name = 'intensity', data = recon_wave.intensity)
        gp.create_dataset(name = 'phase', data = recon_wave.phase)
        return recon_wave
    
    def add_hologram(self, hologram, time_point = 0):
        """
        Add a hologram to the time-series.

        Parameters
        ----------
        hologram : Hologram
        
        time_point : float, optional
        """
        # Extend metadata        
        if time_point not in self.time_points:
            self.time_points = self.time_points + (time_point,)
        
        gp = self._time_group(time_point = time_point)
        gp.create_dataset(name = 'hologram', data = hologram.hologram, 
                          chunks = True, compression = 'lzf', shuffle = True)
    
    # Navigating the HDF5 group using the methods below is preferred since groups will
    # be created on the fly if not in existence.
    def _time_group(self, time_point):
        """
        HDF5 group in which is stored the Hologram, and wavelengths groups.

        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        
        Returns
        -------
        group : `~h5py.Group`
        """
        return self.require_group(str(time_point))
    
    def _reconstruction_group(self, time_point, wavelength):
        """
        HDF5 group in which is stored a hologram reconstruction.

        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        wavelength : int
            Wavelength in nm.
        
        Returns
        -------
        group : `~h5py.Group`
        """
        return self._time_group(time_point).require_group(str(int(wavelength)))