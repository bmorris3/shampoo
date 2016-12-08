"""
This module facilitates storage and acces to holography datasets
and their reconstructions via HDF5.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import h5py
import os
from skimage.io import imread
from astropy.utils.console import ProgressBar

__all__ = ['HDF5Archive']


def tiff_to_ndarray(path):
    """Read in TIFF file, return `~numpy.ndarray`"""
    return np.array(imread(path), dtype=np.float64)


def create_hdf5_archive(hdf5_path, hologram_paths, n_z, metadata={},
                        compression='lzf', overwrite=False):
    """
    Create HDF5 file structure for holograms and phase/intensity
    reconstructions.

    Parameters
    ----------
    hdf5_path : string
        Name of new HDF5 archive
    hologram_paths : string
        List of all holograms
    n_z : int
        Number of z-stacks to allocate space for
    meta : dict
        Metadata to store with in top-level of the HDF5 archive

    Returns
    -------
    f : `~h5py.File`
        Opened HDF5 file
    """
    if os.path.exists(hdf5_path) and not overwrite:
        raise ValueError("File {0} already exists. To overwrite it, "
                         "use `overwrite=True`."
                         .format(hdf5_path))

    f = h5py.File(hdf5_path, 'w')

    first_image = tiff_to_ndarray(hologram_paths[0])

    # Create datasets for holograms, fill it in with holograms, metadata
    f.create_dataset('holograms', dtype=first_image.dtype,
                     shape=(len(hologram_paths),
                            first_image.shape[0], first_image.shape[1]),
                     compression=compression)

    # Update attributes on `holograms` with metadata
    f['holograms'].attrs.update(metadata)

    holograms_dset = f['holograms']
    print('Loading holograms into file {0}...'.format(hdf5_path))
    with ProgressBar(len(hologram_paths)) as bar:
        for i, path in enumerate(hologram_paths):
            holograms_dset[i, :, :] = tiff_to_ndarray(path)
            bar.update()

    # Create empty datasets for reconstructions
    reconstruction_dtype = np.complex128
    f.create_dataset('reconstructed_wavefields', dtype=reconstruction_dtype,
                     shape=(len(hologram_paths), n_z,
                            first_image.shape[0], first_image.shape[1]),
                     compression=compression)
    return f


def open_hdf5_archive(hdf5_path):
    """
    Load and return a shampoo HDF5 archive.

    Parameters
    ----------
    hdf5_path : string
        Name of HDF5 archive

    Returns
    -------
    f : `~h5py.File`
        Opened HDF5 file
    """
    return h5py.File(hdf5_path, 'r+')


def _time_index_to_string(time_index):
    return "t{0:05d}".format(time_index)


def _wavelength_index_to_string(wavelength_index):
    return "wavelength{0:d}".format(wavelength_index)


class HDF5Archive(object):
    def __init__(self, path, overwrite=False):
        self.path = path

        # Create/open the HDF5 file stream
        mode = 'r+' if os.path.exists(path) and not overwrite else 'w'
        self.f = h5py.File(path, mode)
        self.is_open = True

    def reopen(self):
        if not self.is_open:
            self.f = h5py.File(self.path, 'r+')
            self.is_open = True

    def create_group_for_timestep(self, time_index, shape, n_wavelengths=1,
                                  compression='lzf'):
        from .reconstruction import float_precision

        if _time_index_to_string(time_index) not in self.f:
            time_group = self.f.create_group(_time_index_to_string(time_index))

            for i in range(n_wavelengths):
                wl_group = time_group.create_group(_wavelength_index_to_string(i))

                phase_dataset = wl_group.create_dataset('phase',
                                                        dtype=float_precision,
                                                        shape=shape,
                                                        compression=compression)
                intensity_dataset = wl_group.create_dataset('intensity',
                                                            dtype=float_precision,
                                                            shape=shape,
                                                            compression=compression)
                hologram_dataset = wl_group.create_dataset('hologram',
                                                           dtype=float_precision,
                                                           shape=shape,
                                                           compression=compression)

    def create_groups_for_series(self, n_times, shape, n_wavelengths=1,
                                 compression='lzf'):
        for time_index in range(n_times):
            self.create_group_for_timestep(time_index, shape,
                                           n_wavelengths=n_wavelengths,
                                           compression=compression)

    def update(self, data, time_index, distance_index, wavelength_index,
               data_type=None):

        if data_type is None:
            raise ValueError("Must specify data type "
                             "(either phase or intensity)")

        data_type = data_type.lower().strip()

        time = _time_index_to_string(time_index)
        wavelength = _wavelength_index_to_string(wavelength_index)

        self.f[time][wavelength][data_type][distance_index, ...] = data

        self.f.flush()

    def get(self, time_index, distance_index, wavelength_index, data_type=None):

        if data_type is None:
            raise ValueError("Must specify data type "
                             "(either phase or intensity)")

        data_type = data_type.lower().strip()

        time = _time_index_to_string(time_index)
        wavelength = _wavelength_index_to_string(wavelength_index)

        return self.f[time][wavelength][data_type][distance_index, ...][:]

    def close(self):
        self.f.flush()
        self.f.close()
        self.is_open = False