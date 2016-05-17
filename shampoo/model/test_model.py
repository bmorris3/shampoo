from lorenzmie import lmsphere
from astropy.io import fits
import numpy as np

def test_example_hologram():
    py_holo = lmsphere([0, 0, 200], 0.75, 1.5, 1.33, 0.532, 0.135, [201, 201])
    idl_holo = fits.getdata('../../data/idl_example_hologram.fits')

    assert np.mean(np.abs(py_holo - idl_holo)) < 1e-5
