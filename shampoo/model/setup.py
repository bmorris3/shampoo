from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name = "My hello app",
      ext_modules = cythonize('lorenzmie.pyx'),
      include_dirs=[numpy.get_include()])