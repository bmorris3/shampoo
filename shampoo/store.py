"""
This module facilitates storage and acces to holography datasets
and their reconstructions via HDF5.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from PIL import Image
import numpy as np
from glob import glob
import h5py
import os

__all__ = ['create_hdf5']

def tiff_to_ndarray(path):
    """Read in TIFF file, return `~numpy.ndarray`"""
    return np.array(Image.open(path))

def create_archive(hdf5_path, hologram_glob_pattern, n_z, meta={},
                   compression='gzip', overwrite=False):
    """
    Create HDF5 file structure for holograms and phase/intensity
    reconstructions with the following format:

    /
    |--holograms
    |--phase
    |--intensity

    Parameters
    ----------
    hdf5_path : string
        Name of new HDF5 archive
    hologram_glob_pattern : string
        Passed to `~glob.glob` used to collect all holograms
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

    hologram_paths = glob(hologram_glob_pattern)
    first_image = tiff_to_ndarray(hologram_paths[0])

    # Create datasets for holograms, fill it in with holograms, metadata
    f.create_dataset('holograms', dtype=first_image.dtype,
                     shape=(len(hologram_paths),
                            first_image.shape[0], first_image.shape[1]),
                     compression=compression)

    # Update attributes on `holograms` with metadata
    f['holograms'].attrs.update(metadata)

    holograms_dset = f['holograms']
    for i, path in enumerate(hologram_paths):
        holograms_dset[i, :, :] = tiff_to_ndarray(path)

    # Create empty datasets for phase, intensity reconstructions
    reconstruction_dtype = np.float64
    f.create_dataset('phase', dtype=reconstruction_dtype,
                     shape=(len(hologram_paths), n_z,
                            first_image.shape[0], first_image.shape[1]),
                     compression=compression)

    f.create_dataset('intensity', dtype=reconstruction_dtype,
                     shape=(len(hologram_paths), n_z,
                            first_image.shape[0], first_image.shape[1]),
                     compression=compression)

    return f

# def load_archive(hdf5_path):


metadata = dict(date="2015-05-15",
                organism="Vibrio",
                slide=True,
                dye=True)

n_z = 100
f = create_archive("/Volumes/tycho/shamu/complete.hdf5",
                   "/Volumes/tycho/shamu/2015.05.15_17-58_Vibrio_on_slide_+_dye_r/*_holo.tif",
                   n_z, metadata, overwrite=True, compression='gzip')

# def load_hdf5(path):
#     f = h5py.File(path, 'r')
#