from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from multiprocessing.dummy import Pool as ThreadPool

from .reconstruction import _load_hologram

import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft, MexicanHat2DKernel
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from mst_clustering import MSTClustering

# Try importing optional dependency PyFFTW for Fourier transforms. If the import
# fails, import scipy's FFT module instead
try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
except ImportError:
    from scipy.fftpack import fft2, ifft2

__all__ = ["median_hologram", "locate_from_hologram"]


def median_hologram(hologram_paths):
    """
    Take the median of all holograms in a list.

    Be sure only to use holograms with uniform pixel dimensions.

    Parameters
    ----------
    hologram_paths : list of strings
        List of paths to holograms

    Returns
    -------
    hologram_cube : `~numpy.ndarray`
        Cube of all raw holograms
    median_holo : `~numpy.ndarray`
        Median of the holograms in the list
    """
    hologram0 = _load_hologram(hologram_paths[0])
    hologram_cube = np.zeros((len(hologram_paths), hologram0.shape[0],
                              hologram0.shape[1]))

    for i, hologram_path in enumerate(hologram_paths):
        hologram_cube[i, :, :] = _load_hologram(hologram_path)

    median_holo = np.median(hologram_cube, axis=0)

    return hologram_cube, median_holo


def locate_from_hologram(hologram_cube, median_holo, threads=8,
                         convolution_kernel_radius=25,
                         gaussian_kernel_radius=30):
    """
    Find (x, y) positions of specimens on median-subtracted holograms with
    multithreading.

    The procedure for finding specimen positions is as follows. First, median
    subtract each raw hologram, and convolve with the
    `~astropy.convolution.MexicanHat2DKernel` kernel to produce ringing positive
    and negative concentric peaks where specimens are centered. Then smooth the
    absolute value of that convolution to produce smooth positive peaks where
    the specimens are.

    Parameters
    ----------
    hologram_cube : `~numpy.ndarray`
        Cube of all raw holograms
    median_holo : `~numpy.ndarray`
        Median of the holograms in ``hologram_cube``
    threads : int (optional)
        Spin up ``threads`` processes
    convolution_kernel_radius : float (optional)
        Set `~astropy.convolution.MexicanHat2DKernel` radius.
    gaussian_kernel_radius : float (optional)
        Set `~scipy.ndimage.gaussian_filter` smoothing kernel radius.

    Returns
    -------
    positions : `~numpy.ndarray`
        Positions for each detected specimen in each frame, with values
        ``(t, x, y, max_intensity, mean_intensity)``
    """
    n_times = hologram_cube.shape[0]
    positions = []

    def locate(i):
        median_sub_holo = hologram_cube[i, ...] - median_holo
        conv_holo = convolve_fft(median_sub_holo,
                                 MexicanHat2DKernel(convolution_kernel_radius),
                                 fftn=fft2, ifftn=ifft2)
        smooth_abs_conv = gaussian_filter(np.abs(conv_holo),
                                          gaussian_kernel_radius)

        thresh = threshold_otsu(smooth_abs_conv)

        masked = np.ones_like(smooth_abs_conv)
        masked[smooth_abs_conv <= thresh] = 0

        label_image = label(masked)
        regions = regionprops(label_image, smooth_abs_conv)
        for region in regions:
            centroid = region.weighted_centroid
            pos = (i, centroid[0], centroid[1],
                   region.max_intensity, region.mean_intensity)
            positions.append(pos)

    pool = ThreadPool(threads)
    pool.map(locate, range(n_times))

    # close the pool and wait for the work to finish
    pool.close()
    pool.join()

    return np.vstack(positions)


def cluster_positions(positions, plots=False, cutoff_scale_min=120,
                      cutoff_scale_max=350, cutoff_scale_resolution=150,
                      n_neighbors_max=5, min_cluster_size=4):
    """
    Find clusters in the positions produced by
    `~shampoo.track2d.locate_from_hologram` using a Minimum Spanning Tree.

    Parameters
    ----------
    positions : `~numpy.ndarray`
        Positions for each detected specimen in each frame, with values
        specified by `~shampoo.track2d.locate_from_hologram`.
    plots : bool (optional)
        Plot the scores for the grid search in ``(cutoff_scale, n_neighbors)``
        space and the best clusters

    Returns
    -------
    labels : `~numpy.ndarray`
        Cluster labels for each position in ``positions``. `-1` represents
        positions without a cluster.
    """
    # Scale the time and max_intensity dims similarly to the spatial dimensions
    X = positions.copy()
    X = X[:, 0:4]
    X[:, 0] *= 5*positions[:, 1].ptp()/positions[:, 0].ptp()
    X[:, 3] *= positions[:, 1].ptp()/positions[:, 3].ptp()

    # Grid search in (cutoff_scales, n_neighbors) for the best clustering params
    cutoff_scales = np.linspace(cutoff_scale_min, cutoff_scale_max,
                                cutoff_scale_resolution)
    n_neighbors = np.arange(1, n_neighbors_max)
    scores = np.zeros((len(cutoff_scales), len(n_neighbors)), dtype=np.float64)

    for i in range(cutoff_scales.shape[0]):
        for j in range(n_neighbors.shape[0]):
            model = MSTClustering(cutoff_scale=cutoff_scales[i],
                                  approximate=True, n_neighbors=n_neighbors[j],
                                  min_cluster_size=min_cluster_size)
            labels = model.fit_predict(X)

            distance_stds = []
            for l in set(labels):
                if l != -1:
                    pca = PCA(n_components=3)
                    pca.fit(X[labels == l, 0:3])
                    X_prime = pca.transform(X[labels == l, 0:3])

                    distance_stds.append(X_prime[:, 1].std() /
                                         X_prime[:, 0].ptp())

            f_labeled = np.count_nonzero(labels != -1)/float(len(labels))
            scores[i, j] = np.mean(distance_stds)/f_labeled

    # With the best clustering parameters, label the clusters
    x_min_ind, y_min_ind = np.where(scores == scores.min())
    n_neighbors_min = n_neighbors[y_min_ind[0]]
    cuttoff_scale_min = cutoff_scales[x_min_ind[0]]

    model = MSTClustering(cutoff_scale=cuttoff_scale_min, approximate=True,
                          n_neighbors=n_neighbors_min, min_cluster_size=4)
    labels = model.fit_predict(X)

    if plots:
        # Plot the scores in (cutoff_scales, n_neighbors) space
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(np.log(scores).T, interpolation='nearest', origin='lower',
                  cmap=plt.cm.viridis)
        ax.set_xticks(range(len(cutoff_scales))[::5])
        ax.set_xticklabels(["{0:.2f}".format(cutoff_scale)
                            for cutoff_scale in cutoff_scales[::5]])

        ax.set_yticks(range(len(n_neighbors)))
        ax.set_yticklabels(range(1, len(n_neighbors)+1))

        for l in ax.get_xticklabels():
            l.set_rotation(45)
            l.set_ha('right')

        ax.set_xlabel('cutoff')
        ax.set_ylabel('n_neighbors')
        ax.set_aspect(10)

        # Plot the best clusters
        plot_segments = True
        fig, ax = plt.subplots(1, 3, figsize=(16, 6))

        kwargs = dict(s=100, alpha=0.6, edgecolor='none', cmap=plt.cm.Spectral,
                      c=labels)
        ax[0].scatter(X[:, 0], X[:, 1], **kwargs)
        ax[1].scatter(X[:, 0], X[:, 2], **kwargs)
        ax[2].scatter(X[:, 1], X[:, 2], **kwargs)

        ax[0].set(xlabel='t', ylabel='x')
        ax[1].set(xlabel='t', ylabel='y')
        ax[2].set(xlabel='x', ylabel='y')

        if plot_segments:
            segments = model.get_graph_segments(full_graph=False)
            ax[0].plot(segments[0], segments[1], '-k')
            ax[1].plot(segments[0], segments[2], '-k')
            ax[2].plot(segments[1], segments[2], '-k')

        fig.tight_layout()

        plt.show()

    return labels
