"""
This module facilitates storage and acces to holography datasets
and their reconstructions via HDF5.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from PIL import Image
import numpy as np
import h5py
import os
from astropy.utils.console import ProgressBar

__all__ = ['create_hdf5_archive', 'open_hdf5_archive']

def tiff_to_ndarray(path):
    """Read in TIFF file, return `~numpy.ndarray`"""
    return np.array(Image.open(path))

def create_hdf5_archive(hdf5_path, hologram_paths, n_z, metadata={},
                        compression='lzf', overwrite=False):
    """
    Create HDF5 file structure for holograms and phase/intensity
    reconstructions with the following format:

    /
    |--holograms
    |--reconstructed_wavefields

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
    Load and return a shampoo HDF5 archive

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

