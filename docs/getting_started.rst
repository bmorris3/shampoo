.. include:: references.txt

.. _getting_started:

***************
Getting Started
***************


Simple numerical reconstruction
===============================

shampoo comes with a sample hologram of a US Air Force resolution target
to reconstruct in the ``data`` directory. Let's reconstruct it with shampoo.

Let's start in the shampoo repository's top level directory. First, one must
import shampoo, and specify the path to the file to reconstruct and the
propagation distance::

    from shampoo import Hologram
    hologram_path = 'data/USAF_test.tif'
    propagation_distance = 0.03685  # m

Now we will import the hologram, which is stored in TIF format, using the
`~shampoo.Hologram.from_tif` method::


    h = Hologram.from_tif(hologram_path)

If your digital holographic microscope has different properties from SHAMU, for
example, a different wavelength and pixel size, you will want to set those
properties in the `~shampoo.Hologram` constructor, like this::

    h = Hologram.from_tif(hologram_path, wavelength=650e-6, dx=3.5e-6, dy=3.5e-6)

The defaults are set for SHAMU's configuration, so the defaults will work
for this example hologram which was recorded by SHAMU.

Now let's reconstruct the hologram::

    wave = h.reconstruct(propagation_distance)

That's it! The reconstructed wave is stored within ``wave``, which is a
`~shampoo.ReconstructedWave` object that stores the 2D complex `~numpy.ndarray`
reconstructed wave, plus has attributes to get the phase and intensity arrays
(`~shampoo.ReconstructedWave.phase` and `~shampoo.ReconstructedWave.intensity`),
and a plotting function `~shampoo.ReconstructedWave.plot` which we will use to
plot the phase and intensity from the reconstructed wave::

    import matplotlib.pyplot as plt
    fig, ax = wave.plot()
    fig.suptitle("USAF Target")
    fig.tight_layout()
    plt.show()

Here's the result:

.. plot::

    # Import package, set hologram path, propagation distance
    from shampoo import Hologram
    hologram_path = '../data/USAF_test.tif'
    propagation_distance = 0.03685  # m

    # Construct the hologram object, reconstruct the complex wave
    h = Hologram.from_tif(hologram_path)
    wave = h.reconstruct(propagation_distance)

    # Plot the reconstructed phase/intensity
    import matplotlib.pyplot as plt
    fig, ax = wave.plot()
    fig.suptitle("USAF Target")
    fig.tight_layout()
    plt.show()

Now you're ready to reconstruct your holograms!

.. warning::

    This release should be considered a "preview", as shampoo is still under
    development.

For more tutorials, see the Tutorials documentation.

:ref:`Return to Top <getting_started>`