from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from skimage.io import imsave

__all__ = ['glue_focus', 'save_scaled_image']


def glue_focus(xyz, labels):
    """
    Launch a glue session with focusing results.

    Parameters
    ----------
    xyz : `~numpy.ndarray`
        Matrix of (x, y, z) positions of particles
    labels : `~numpy.ndarray`
        Labels for particles assigned by clustering algorithm

    Returns
    -------
    ga : `~glue.app.qt.application.GlueApplication`
        Glue qt GUI application session
    """
    from glue.core import DataCollection, Data
    from glue.app.qt.application import GlueApplication
    from glue_vispy_viewers.scatter.scatter_viewer import VispyScatterViewer

    data = Data(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                clusters=labels, label='data')
    dc = DataCollection([data])

    # create a GUI session
    ga = GlueApplication(dc)
    scatter = ga.new_data_viewer(VispyScatterViewer)
    scatter.add_data(data)

    return ga


def save_scaled_image(image, filename, margin=100, blobs=None,
                      min=0.01, max=99.99):  #, min=0.05, max=99.95):
    """
    Save an image to png.

    Parameters
    ----------
    image : `~numpy.ndarray`
        Image to save
    filename : str
        Path to where to save the png file
    blobs : list or `~numpy.ndarray` or `None`
        (x, y, z) positions
    margin : 
    min : float
        Colormap scaling minimum
    max : float
        Colormap scaling maximum
    """
    img_scaled = image.copy()

    if img_scaled.shape[0] > 1000:
        scale_margin = 200
    elif img_scaled.shape[0] > 500:
        scale_margin = 50
    elif img_scaled.shape[0] < 100:
        scale_margin = 0
    else:
        scale_margin = 10

    if img_scaled.shape[0] > 100:
        center_stamp = image[scale_margin:-scale_margin]
        img_scaled[np.percentile(center_stamp, min) > image] = np.percentile(center_stamp, min)
        img_scaled[np.percentile(center_stamp, max) < image] = np.percentile(center_stamp, max)

    img_scaled = ((img_scaled - img_scaled.min()) /
                  (img_scaled.max()-img_scaled.min()))

    if blobs is not None:
        for blob in blobs:
            x, y = blob[0] + margin, blob[1] + margin
            lo = 10
            hi = 20
            thick = 2.0
            img_scaled[x+lo:x+hi, y-thick:y+thick] = 1.0
            img_scaled[x-thick:x+thick, y+lo:y+hi] = 1.0
            img_scaled[x-hi:x-lo, y-thick:y+thick] = 1.0
            img_scaled[x-thick:x+thick, y-hi:y-lo] = 1.0

    imsave(filename, img_scaled)

