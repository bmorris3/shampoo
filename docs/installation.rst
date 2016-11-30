.. include:: references.txt

.. _installation:

************
Installation
************

Requirements
============

.. note::

    Users are strongly recommended to manage these dependencies with the
    excellent `Anaconda Python Distribution <http://continuum.io/downloads>`_
    which provides easy access to all of the above dependencies and more.

**shampoo** works on Linux, Mac OS X and Windows. It requires Python 3.5 or 2.7
(earlier versions are not supported) as well as the following packages:

* `Numpy`_
* `scipy`_
* `Matplotlib`_
* `skimage`_
* `sklearn`_
* `Astropy`_
* `h5py`_
* `pyfftw`_

pyFFTW
~~~~~~

shampoo depends on a package called `pyfftw`_ for speedy, multithreaded
Fourier transforms, which is easy to install on Mac OS X and linux but may be
tricky on Windows machines. We recommend that Windows users install pyfftw by
doing the following steps via conda::

    conda install -c salilab fftw
    pip install pyfftw

Install shampoo
===============

You can install the latest developer version of shampoo by cloning the git
repository::

    git clone https://github.com/bmorris3/shampoo.git

...then installing the package with::

    cd shampoo
    python setup.py install


Testing
=======

If you want to check that all the tests are running correctly with your Python
configuration, start up python, and type::

    import shampoo
    shampoo.test()

If there are no errors, you are good to go!

More
~~~~

shampoo follows `Astropy`_'s guidelines for affiliated packages--installation
and testing for the two are quite similar! Please see Astropy's
`installation page <http://astropy.readthedocs.org/en/latest/install.html>`_
for more information.
