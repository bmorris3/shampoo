from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from glue.core import DataCollection, Data
from glue.app.qt.application import GlueApplication

__all__ = ['glue_focus']


def glue_focus(xyz, labels):
    """
    Launch a glue session with focusing results.

    Parameters
    ----------
    xyz : `~numpy.ndarray`
        Matrix of (x, y, z) positions of particles
    labels : `~numpy.ndarray`
        Labels for particles assigned by clustering algorithm
    """
    from glue_vispy_viewers.scatter.scatter_viewer import VispyScatterViewer

    data = Data(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                l=labels, label='data')
    dc = DataCollection([data])

    # create a GUI session
    ga = GlueApplication(dc)
    scatter = ga.new_data_viewer(VispyScatterViewer)
    scatter.add_data(data)

    ga.start()

