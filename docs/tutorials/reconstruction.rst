.. include:: ../references.txt

.. _reconstruction_tutorial:

************************
Reconstruction Tutorials
************************

If you're looking for a quick reference for beginning your first reconstruction,
see :ref:`Getting Started <getting_started>`. Below we'll dive into some more
in-depth examples.

Contents
========

* :ref:`reconstruction_tutorial-multiple_zs`
* :ref:`reconstruction_tutorial-crop`
* :ref:`reconstruction_tutorial-multithreading`

.. _reconstruction_tutorial-multiple_zs:

Reconstructing multiple z-planes
================================

If you want to reconstruct several z-planes – for example, to find the
propagation distance to the USAF test target – you can use the following
pattern. First, we will load the raw hologram (only once)::

    from shampoo import Hologram
    import numpy as np

    hologram_path = 'data/USAF_test.tif'
    h = Hologram.from_tif(hologram_path)

Next we must set the range of propagation distances to reconstruct. We will
use `~numpy.linspace` to create a linearly-spaced array of five propagation
distances::

    n_z_slices = 5
    propagation_distances = np.linspace(0.03585, 0.03785, n_z_slices)

Then we will loop over the propagation distances, calling
`~shampoo.Hologram.reconstruct` for each one. We will store the reconstructed
wave intensities into a data cube called ``intensities``::

    # Allocate some memory for the complex reconstructed waves
    intensities = np.zeros((n_z_slices, h.hologram.shape[0], h.hologram.shape[1]),
                           dtype=np.complex128)

    # Loop over all propagation distances
    for i, distance in enumerate(propagation_distances):

        # Reconstruct at each distance
        wave = h.reconstruct(distance)

        # Save the result to the data cube
        intensities[i, ...] = wave.intensity

Now ``intensities`` contains all of the reconstructed wave intensities, so if we
want to find the propagation distance at best focus, we could apply a very crude
focus metric to the intensity arrays. For example, the standard deviation of the
intensities will be maximal at the propagation distance nearest to the USAF
target, so let's plot the standard deviation of the intensities at each
propagation distance::

    # Measure standard deviation within each intensity image
    std_intensities = np.std(intensities, axis=(1, 2))

    # Initialize a figure object
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(propagation_distances, std_intensities, 'o-')
    ax.axvline(propagation_distances[np.argmax(std_intensities)], ls='--',
               label='Best propagation distance')
    ax.set(xlabel='Propagation distance', ylabel='std( Intensity )')
    ax.legend()
    plt.show()

.. plot::

    from shampoo import Hologram
    import numpy as np

    hologram_path = '../../data/USAF_test.tif'
    h = Hologram.from_tif(hologram_path)

    n_z_slices = 5
    propagation_distances = np.linspace(0.03585, 0.03785, n_z_slices)

    # Allocate some memory for the reconstructed intensity images
    intensities = np.zeros((n_z_slices, h.hologram.shape[0], h.hologram.shape[1]),
                           dtype=np.complex128)

    # Loop over all propagation distances
    for i, distance in enumerate(propagation_distances):

        # Reconstruct at each distance
        wave = h.reconstruct(distance)

        # Save the result to the data cube
        intensities[i, ...] = wave.intensity

    # Measure standard deviation within each intensity image
    std_intensities = np.std(intensities, axis=(1, 2))

    # Initialize a figure object
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(propagation_distances, std_intensities, 'o-')
    ax.axvline(propagation_distances[np.argmax(std_intensities)], ls='--',
               label='Best propagation distance')
    ax.set(xlabel='Propagation distance', ylabel='std( Intensity )')
    ax.legend()
    plt.show()

You can see that the best propagation distance is near the same distance that we
used in the first example (that's why we picked it!).

:ref:`Return to Top <reconstruction_tutorial>`



.. _reconstruction_tutorial-crop:

Reconstructing a cropped hologram
=================================

Sometimes you need speed. Sometimes you can afford to reconstruct only the
central part the hologram's your field of view if it will save you some
time. `~shampoo.Hologram` has a built-in option to help you in this situation.

The keyword argument ``crop_fraction`` sets the fraction of the original image
to crop out. By default it is set to `None`. If you set ``crop_fraction=0.5``,
a 1024x1024 hologram will be cropped to 512x512 before being reconstructed::

    # Import package, set hologram path, propagation distance
    from shampoo import Hologram
    hologram_path = 'data/USAF_test.tif'
    propagation_distance = 0.03685  # m

    # Construct the hologram object, reconstruct the complex wave
    h = Hologram.from_tif(hologram_path, crop_fraction=0.5)
    wave = h.reconstruct(propagation_distance)

    # Plot the reconstructed phase/intensity
    import matplotlib.pyplot as plt
    fig, ax = wave.plot()
    fig.suptitle("USAF Target")
    fig.tight_layout()
    plt.show()


Since the implementations of the FFT used by shampoo generally run on order
:math:`O(N log N)` time, the cropped hologram reconstruction should compute
about 2.2x faster than the full-sized one.

.. plot::

    # Import package, set hologram path, propagation distance
    from shampoo import Hologram
    hologram_path = '../../data/USAF_test.tif'
    propagation_distance = 0.03685  # m

    # Construct the hologram object, reconstruct the complex wave
    h = Hologram.from_tif(hologram_path, crop_fraction=1/2)
    wave = h.reconstruct(propagation_distance)

    # Plot the reconstructed phase/intensity
    import matplotlib.pyplot as plt
    fig, ax = wave.plot()
    fig.suptitle("USAF Target")
    fig.tight_layout()
    plt.show()

.. _reconstruction_tutorial-multithreading:

Reconstruction with multithreading
==================================

For an "embarrassingly parallel" version of the reconstruction method, which is
useful for quickly reconstructing many z-planes of the same hologram, there's
an alternative to `~shampoo.Hologram.reconstruct` called
`~shampoo.Hologram.reconstruct_multithread`. This method takes an array or list
of propagation distances rather than a single distance, and the keyword
``threads`` which sets the number of threads to spawn in the
`~multiprocessing.dummy.Pool`. This should be a number similar to the number of
cores in your CPU.

Let's set up a similar reconstruction to the one in the first example above,
where we'll reconstruct the USAF target five times::

    import numpy as np

    hologram_path = '../../data/USAF_test.tif'
    h = Hologram.from_tif(hologram_path, crop_fraction=0.5)

    n_z_slices = 5
    propagation_distances = np.linspace(0.03585, 0.03785, n_z_slices)

This time, we'll call `~shampoo.Hologram.reconstruct_multithread`, and we'll
issue this task to four threads::

    # Reconstruct at each distance
    waves = h.reconstruct_multithread(propagation_distances, threads=4)

The output ``waves`` is now a three-dimensional, complex `~numpy.ndarray` of the
reconstructed waves with shape :math:`(N_z, X, Y)` where :math:`N_z` is the
number of propagation distances, and :math:`(X, Y)` are the pixel dimensions of
the reconstructed wave.

Now we can perform operations on this 3D array of reconstructed waves easily.
For example, let's see if the sum of the intensity in each wave is a proxy for
the focus – an alternative to the standard deviation used in the previous
example. The standard deviation works as a focus metric for images with sharp
discontinuities, like the USAF target has. The minimum in the sum of the
intensity should also work as a focus metric for targets that mostly absorb
light, which the USAF target does. Let's sum those intensities and plot
the result::

    # Allocate some memory for the complex reconstructed waves
    intensities = np.zeros((n_z_slices, h.hologram.shape[0], h.hologram.shape[1]),
                           dtype=np.complex128)

    # Loop over all propagation distances
    for i, distance in enumerate(propagation_distances):

        # Reconstruct at each distance
        intensities[i, ...] = np.abs(waves[i, ...])

    # Measure standard deviation within each intensity image
    sum_intensities = np.sum(intensities, axis=(1, 2))

    # Initialize a figure object
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(propagation_distances, sum_intensities, 'o-')
    ax.axvline(propagation_distances[np.argmin(sum_intensities)], ls='--',
               label='Best propagation distance')
    ax.set(xlabel='Propagation distance', ylabel='sum( Intensity )')
    ax.legend()
    plt.show()


The minimum in the sum of the intensities is found near propagation
distance as we found in the early example. The result is slightly different
because we have used the ``crop_fraction`` keyword to reconstruct only a
fraction of the hologram (to save build time on the Read The Docs server).

.. plot::

    from shampoo import Hologram
    import numpy as np

    hologram_path = '../../data/USAF_test.tif'
    h = Hologram.from_tif(hologram_path, crop_fraction=0.5)

    n_z_slices = 5
    propagation_distances = np.linspace(0.03585, 0.03785, n_z_slices)

    # Reconstruct at each distance
    waves = h.reconstruct_multithread(propagation_distances, threads=4)

    # Allocate some memory for the complex reconstructed waves
    intensities = np.zeros((n_z_slices, h.hologram.shape[0], h.hologram.shape[1]),
                           dtype=np.complex128)

    # Loop over all propagation distances
    for i, distance in enumerate(propagation_distances):

        # Reconstruct at each distance
        intensities[i, ...] = np.abs(waves[i, ...])

    # Measure standard deviation within each intensity image
    sum_intensities = np.sum(intensities, axis=(1, 2))

    # Initialize a figure object
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(propagation_distances, sum_intensities, 'o-')
    ax.axvline(propagation_distances[np.argmin(sum_intensities)], ls='--',
               label='Best propagation distance')
    ax.set(xlabel='Propagation distance', ylabel='sum( Intensity )')
    ax.legend()
    plt.show()

:ref:`Return to Top <reconstruction_tutorial>`

