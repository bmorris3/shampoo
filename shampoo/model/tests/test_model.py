from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.io import fits
import numpy as np
from ..lorenzmie import lorenz_mie_field_cartesian
from astropy.tests.helper import remote_data
from astropy.utils.data import download_file

IDL_HOLOGRAM_URL = 'http://staff.washington.edu/bmmorris/images/idl_example_hologram.fits'

@remote_data
def test_example_hologram():
    py_holo = lorenz_mie_field_cartesian([0, 0, 200], 0.75, 1.5, 1.33, 0.532,
                                         0.135, [201, 201])
    idl_holo = fits.getdata(download_file(IDL_HOLOGRAM_URL))

    assert np.mean(np.abs(py_holo - idl_holo)) < 1e-5

