#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=True
#filename: lorenzmie.pyx

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from libc.math cimport sin, cos, exp, sqrt
cimport cython

cimport numpy as np
DTYPE_COMPLEX = np.complex128
ctypedef np.complex128_t DTYPE_COMPLEX_t
DTYPE_FLOAT = np.float64
ctypedef np.float64_t DTYPE_FLOAT_t

def sin_1d(np.ndarray[DTYPE_FLOAT_t, ndim=1] array):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] result = np.zeros(array.shape[0], dtype=DTYPE_FLOAT)
    cdef unsigned int i
    cdef int n = array.shape[0]
    for i in range(n):
        result[i] = sin(array[i])
    return result

def cos_1d(np.ndarray[DTYPE_FLOAT_t, ndim=1] array):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] result = np.zeros(array.shape[0], dtype=DTYPE_FLOAT)
    cdef unsigned int i
    cdef int n = array.shape[0]
    for i in range(n):
        result[i] = cos(array[i])
    return result

def sin_2d(np.ndarray[DTYPE_FLOAT_t, ndim=2] array):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] result = np.zeros_like(array, dtype=DTYPE_FLOAT)
    cdef unsigned int i
    cdef int n0 = array.shape[0]
    cdef int n1 = array.shape[1]
    for i in range(n0):
        for j in range(n1):
            result[i, j] = sin(array[i, j])
    return result

def exp_1d(np.ndarray[DTYPE_FLOAT_t, ndim=1] array):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] result = np.zeros(array.shape[0], dtype=DTYPE_FLOAT)
    cdef unsigned int i
    cdef int n0 = array.shape[0]
    for i in range(n0):
        result[i] = exp(array[i])
    return result

def quadrature_sum(np.ndarray[DTYPE_FLOAT_t, ndim=2] x, np.ndarray[DTYPE_FLOAT_t, ndim=2] y):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] result = np.zeros_like(x, dtype=DTYPE_FLOAT)
    cdef int n0 = x.shape[0]
    cdef int n1 = x.shape[1]
    cdef unsigned int i, j
    for i in range(n0):
        for j in range(n1):
            result[i, j] = sqrt(x[i, j]*x[i, j] + y[i, j]*y[i, j])
    return result

def positive_roll(np.ndarray[DTYPE_COMPLEX_t, ndim=1] a, int steps):
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=1] b = np.zeros_like(a, dtype=DTYPE_COMPLEX)
    cdef int len_a = b.shape[0]
    cdef unsigned int i
    cdef int offset
    for i in range(len_a):
        if i >= steps:
            b[i] = a[<unsigned int>(i-steps)]
        if i < steps:
            offset = b.shape[0] - steps
            b[i] = a[<unsigned int>(i + offset)]
    return b

def arange(int n, float offset):
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] b = np.zeros(n, dtype=DTYPE_FLOAT)
    cdef unsigned int i
    for i in range(n):
        b[i] = i + offset
    return b

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def lmsphere(list rp, float ap, complex n_sphere, complex nm, float lambda_,
             float mpp, list dim, alpha=1, delta=0, precision=None):
    """
    rp  : [x,y,z] 3 dimensional position of sphere relative to lower-left corner of image.
    ap  : radius of sphere [micrometers]
    n_sphere  : (complex) refractive index of sphere
    nm  : (complex) refractive index of medium
    lambda: vacuum wavenp.sizegth of light [micrometers]
    mpp: micrometers per pixel
    dim : [nx,ny] dimensions of image [pixels]

    alpha: fraction of incident light scattered by particle.
       Default: 1.

    delta: wavefront distortion [wavenp.sizegths]
           Default: 0.

    field: On output, field in imaging plane

    precision: relative precision with which fields are calculated.
    """
    cdef int nx = dim[0]
    cdef int ny = dim[1]
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] x_i = arange(nx, offset=-rp[0])
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] y_i = arange(ny, offset=-rp[1])
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] x
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] y
    x, y = np.meshgrid(x_i, y_i)
    cdef float zp = rp[2]

    #field = spherefield(x, y, zp, ap, n_sphere, nm, lambda_, mpp)

    #cdef np.ndarray ab = np.zeros()
    cdef complex m = n_sphere/nm + 0*1j
    cdef int nmax = Nstop(rp[0], m)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] ab = np.zeros((2, nmax+1), dtype=DTYPE_COMPLEX)
    ab = sphere_coefficients(ap, n_sphere, nm, lambda_)
    cdef float lambda_m = lambda_ / nm.real / mpp ## medium wavenp.sizegth [pixel]

    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] field
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] a
    field = sphericalfield(x, y, zp, ab, lambda_m)# , cartesian=cartesian)

    # BM: copied to this function from spherefield
    cdef float k = 2*np.pi / lambda_m

    field *= alpha * np.exp(-k * (zp + delta) * 1j) ## scattered field
    field[0, :] += 1.                                   ## incident field
    a = np.sum(np.real(field * np.conj(field)), 0)        ## intensity

    return a.reshape((int(nx), int(ny)))

@cython.boundscheck(False) # turn of bounds-checking for entire function
def sphericalfield(np.ndarray[DTYPE_FLOAT_t, ndim=2] x_,
                   np.ndarray[DTYPE_FLOAT_t, ndim=2] y_,
                   float z_,
                   np.ndarray[DTYPE_COMPLEX_t, ndim=2] ab,
                   float lambda_):
    """
    x : [npts] array of pixel coordinates [pixels]
    y : [npts] array of pixel coordinates [pixels]
    z : If field is required in a single plane, then z is the plane's
       distance from the sphere's center [pixels].
       Otherwise, z is an [npts] array of coordinates.

    NOTE: Ideally, x, y and z should be double precision.
          This is left to the calling program for efficiency.

    ab : [2,nc] array of Lorenz-Mie scattering coefficients, where
          nc is the number of terms required for convergence.

    lambda : wavelenth of light in medium [pixels]
    """
    cdef int npts = np.size(x_)
    # nc = np.size(ab[0,:])-1      # number of terms required for convergence
    cdef int nc = np.size(ab[0, :])-1      # number of terms required for convergence

    cdef float k = 2 * np.pi / lambda_         # wavenumber in medium [pixel**-1]

    cdef complex ci = 0 + 1*1j

    # convert to spherical coordinates centered on the sphere.
    # (r, theta, phi) is the spherical coordinate of the pixel
    # at (x,y) in the imaging plane at distance z from the
    # center of the sphere.
    #cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] rho = np.sqrt(x_**2 + y_**2)
    #cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] r = np.sqrt(rho**2 + z_**2)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] rho = quadrature_sum(x_, y_)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] z_array = np.zeros_like(x_, dtype=DTYPE_FLOAT) + z_
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] r = quadrature_sum(rho, z_array)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] theta = np.arctan2(rho, z_)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] phi = np.arctan2(y_, x_)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] costheta = cos_1d(theta.ravel())
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] sintheta = sin_1d(theta.ravel())
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] cosphi = cos_1d(phi.ravel())
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] sinphi = sin_1d(phi.ravel())

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] kr = k*r.ravel()   # reduced radial coordinate

    # starting points for recursive function evaluation ...
    # ... Riccati-Bessel radial functions, page 478
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] sinkr = sin_1d(kr)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] coskr = cos_1d(kr)
    # xi_nm2 = dcomplex(coskr, sinkr) # \xi_{-1}(kr)
    # xi_nm1 = dcomplex(sinkr,-coskr) # \xi_0(kr)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=1] xi_nm2 = coskr + 1j*sinkr # \xi_{-1}(kr)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=1] xi_nm1 = sinkr - 1j*coskr # \xi_0(kr)

    # ... angular functions (4.47), page 95
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] pi_nm1 = np.zeros(npts, dtype=DTYPE_FLOAT)                   # \pi_0(\cos\theta)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] pi_n   = np.ones(npts, dtype=DTYPE_FLOAT)                    # \pi_1(\cos\theta)

    # storage for vector spherical harmonics: [r,theta,phi]
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] Mo1n = np.zeros((3, npts), dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] Ne1n = np.zeros((3, npts), dtype=DTYPE_COMPLEX)

    # storage for scattered field
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] Es = np.zeros((3, npts), dtype=DTYPE_COMPLEX)

    # Compute field by summing multipole contributions
    cdef unsigned int n
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] swisc = np.zeros(npts, dtype=DTYPE_FLOAT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] twisc = np.zeros(npts, dtype=DTYPE_FLOAT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] tau_n = np.zeros(npts, dtype=DTYPE_FLOAT)
    # cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] swisc, twisc, tau_n
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=1] xi_n = np.zeros(npts, dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] Ec
    cdef complex En
    cdef unsigned int m, ii

    cdef int ncols_Es = Es.shape[1]
    cdef int nrows_Es = Es.shape[0]

    for n in range(1, nc):

        for m in range(Es.shape[1]):
            # upward recurrences ...
            # ... Legendre factor (4.47)
            # Method described by Wiscombe (1980)
            swisc[m] = pi_n[m] * costheta[m]
            twisc[m] = swisc[m] - pi_nm1[m]
            tau_n[m] = pi_nm1[m] - n * twisc[m]  # -\tau_n(\cos\theta)

            # ... Riccati-Bessel function, page 478
            xi_n[m] = (2*n - 1) * xi_nm1[m] / kr[m] - xi_nm2[m]    # \xi_n(kr)

            # vector spherical harmonics (4.50)
            Mo1n[1,m] = pi_n[m] * xi_n[m]     # ... divided by cosphi/kr
            Mo1n[2,m] = tau_n[m] * xi_n[m]    # ... divided by sinphi/kr

            dn = (n * xi_n[m])/kr[m] - xi_nm1[m]
            Ne1n[0,m] = n*(n + 1) * pi_n[m] * xi_n[m] # ... divided by cosphi sintheta/kr**2
            Ne1n[1,m] = tau_n[m] * dn      # ... divided by cosphi/kr
            Ne1n[2,m] = pi_n[m]  * dn      # ... divided by sinphi/kr

            # prefactor, page 93
            En = ci**n * (2*n + 1) / n / (n + 1)
            # the scattered field in spherical coordinates (4.45)

            #Es += (En * ci * ab[0,n]) * Ne1n - (En * ab[1,n]) * Mo1n

            for ii in range(Es.shape[0]):
                Es[ii, m] = Es[ii, m] + (En * ci * ab[0,n]) * Ne1n[ii, m] - (En * ab[1,n]) * Mo1n[ii, m]

            # upward recurrences ...
            # ... angular functions (4.47)
            # Method described by Wiscombe (1980)
            pi_nm1[m] = pi_n[m]
            pi_n[m] = swisc[m] + ((n + 1) / n) * twisc[m]

            # ... Riccati-Bessel function
            xi_nm2[m] = xi_nm1[m]
            xi_nm1[m] = xi_n[m]

    # geometric factors were divided out of the vector
    # spherical harmonics for accuracy and efficiency ...
    # ... put them back at the end.
    for n in range(ncols_Es):
        Es[0,n] = Es[0,n] * cosphi[n] * sintheta[n] / kr[n]**2
        Es[1,n] = Es[1,n] * cosphi[n] / kr[n]
        Es[2,n] = Es[2,n] * sinphi[n] / kr[n]

    # Es[0,:] *= cosphi * sintheta / kr**2
    # Es[1,:] *= cosphi / kr
    # Es[2,:] *= sinphi / kr

    # in IDL version by default the scattered wave is returned in spherical
    # coordinates.  Project components onto Cartesian coordinates.
    # Assumes that the incident wave propagates along z and
    # is linearly polarized along x

    # In python version, do cartesian always
    Ec = Es.copy()

    cdef int ncols_Ec = Ec.shape[1]
    for n in range(ncols_Ec):
        Ec[0, n] = Es[0, n] * sintheta[n] * cosphi[n]
        Ec[0, n] = Ec[0, n] + Es[1, n] * costheta[n] * cosphi[n]
        Ec[0, n] = Ec[0, n] - Es[2, n] * sinphi[n]

        Ec[1, n] = Es[0, n] * sintheta[n] * sinphi[n]
        Ec[1, n] = Ec[1, n] + Es[1, n] * costheta[n] * sinphi[n]
        Ec[1, n] = Ec[1, n] + Es[2, n] * cosphi[n]

        Ec[2, n] =  Es[0, n] * costheta[n] - Es[1, n] * sintheta[n]
    return Ec

# def shift(np.ndarray a, int shift):
#     cdef np.ndarray r = np.roll(a, shift)
#     return r
@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def Nstop(int x, complex m):
    """
    Number of terms to keep in the partial wave expansion
    """
    #;; Wiscombe (1980)
    cdef int xl = x
    cdef float ns
    if xl < 8.:
        ns = np.floor(xl + 4. * xl**(1./3.) + 1.)
    elif xl < 4200.:
        ns = np.floor(xl + 4.05 * xl**(1./3.) + 2.)
    elif xl > 4199.:
        ns = np.floor(xl + 4. * xl**(1./3.) + 2.)

    #;; Yang (2003) Eq. (30)
    cdef float choice1, choice2, choice3
    cdef int result
    choice1 = ns
    choice2 = abs(x*m)
    choice3 = np.abs(x*m)
    max_choice = max([choice1, choice2, choice3])
    result = int(np.floor(max_choice) + 15)
    return result

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def sphere_coefficients(float ap, complex n_sphere, complex nm, float lambda_):
    """
    ap : [nlayers] radii of layered sphere [micrometers]
        NOTE: ap and np are reordered automatically so that
        ap is in ascending order.

    n_sphere : [nlayers] (complex) refractive indexes of sphere's layers

    nm : (complex) refractive index of medium

    lambda_ : wavelength of light [micrometers]
    """
    # if not hasattr(ap, '__np.size__'):
    #     ap = np.array([ap])
    # if not hasattr(n_sphere, '__np.size__'):
    #     n_sphere = np.array([n_sphere])

    cdef int nlayers = 1 #np.size(ap)

    cdef float x = 2 * np.pi * np.real(nm) * ap / lambda_ # size parameter [array]
    cdef complex m = n_sphere/nm + 0*1j                    # relative refractive index [array]
    cdef int nmax = Nstop(x, m)              # number of terms in partial-wave expansion
    cdef complex ci = 0 + 1*1j             # imaginary unit

    # arrays for storing results
    #cdef np.ndarray ab, D1, D1_a, D3, D3_a, Psi, Zeta, PsiZeta, PsiZeta_a, Q, Ha, Hb
    # ab = dcomplexarr2d(2, nmax + 1)
    #
    # D1     = dcomplexarr1d(nmax + 2)
    # D1_a   = dcomplexarr2d(nlayers, nmax + 2)
    # # D1_am1 = dcomplexarr2d(nlayers, nmax + 2)
    #
    # D3     = dcomplexarr1d(nmax + 1)
    # D3_a   = dcomplexarr2d(nlayers, nmax + 1)
    # # D3_am1 = dcomplexarr2d(nlayers, nmax + 1)
    #
    # Psi         = dcomplexarr1d(nmax + 1)
    # Zeta        = dcomplexarr1d(nmax + 1)
    # PsiZeta     = dcomplexarr1d(nmax + 1)
    # PsiZeta_a   = dcomplexarr2d(nlayers, nmax + 1)
    # # PsiZeta_am1 = dcomplexarr2d(nlayers, nmax + 1)
    #
    # Q  = dcomplexarr2d(nlayers, nmax + 1)
    # Ha = dcomplexarr2d(nlayers, nmax + 1)
    # Hb = dcomplexarr2d(nlayers, nmax + 1)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] ab = np.zeros((2, nmax + 1), dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=1] D1 = np.zeros(nmax + 2, dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] D1_a = np.zeros((nlayers, nmax + 2), dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=1] D3 = np.zeros(nmax + 1, dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] D3_a = np.zeros((nlayers, nmax + 1), dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=1] Psi = np.zeros(nmax + 1, dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=1] Zeta = np.zeros(nmax + 1, dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=1] PsiZeta = np.zeros(nmax + 1, dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] PsiZeta_a = np.zeros((nlayers, nmax + 1), dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] Q = np.zeros((nlayers, nmax + 1), dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] Ha = np.zeros((nlayers, nmax + 1), dtype=DTYPE_COMPLEX)
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=2] Hb = np.zeros((nlayers, nmax + 1), dtype=DTYPE_COMPLEX)
    cdef float e = np.e

    # Calculate D1, D3 and PsiZeta for Z1 in the first layer
    #z1 = x[0] * m[0]
    cdef complex z1 = x * m
    cdef unsigned int n

    #for n in range(1, nmax+1)[::-1]:
    for n in reversed(range(1, nmax+1)):
        D1_a[0, <unsigned int>(n-1):n] = n/z1 - 1/(D1_a[0, n] + n/z1)

    PsiZeta_a[0, 0:1] = 0.5 * (1 - e**(2 * ci * z1)) # Eq. (18a)

    D3_a[0, 0] = ci                                     # Eq. (18a)
    for n in range(1, nmax):      #upward recurrence Eq. (18b)
        PsiZeta_a[n:<unsigned int>(n+1), 0] = (PsiZeta_a[0, <unsigned int>(n-1)] *
                           (n/z1 - D1_a[0, <unsigned int>(n-1)]) * (n/z1 - D3_a[0, <unsigned int>(n-1)]))
        #D3_a[n:n+1, 0] = D1_a[0, n] + ci/PsiZeta_a[0, n]

    # Ha and Hb in the core
    # Ha[0, :] = D1_a[0, :-1]     # Eq. (7a)
    # Hb[0, :] = D1_a[0, :-1]     # Eq. (8a)
    Ha[0, :] = D1_a[0, :<unsigned int>(D1_a.shape[1]-1)]     # Eq. (7a)
    Hb[0, :] = D1_a[0, :<unsigned int>(D1_a.shape[1]-1)]     # Eq. (8a)


    # z1 = dcomplex(x[-1])
    z1 = x + 0*1j
    # Downward recurrence for D1, Eqs. (16a) and (16b)
    # for n in range(1, nmax)[::-1]: # Eq. (16b)
    for n in reversed(range(1, nmax)): # Eq. (16b)
        D1[<unsigned int>(n-1)] = n/z1 - (1/(D1[n] + n/z1))

    # Upward recurrence for Psi, Zeta, PsiZeta and D3, Eqs. (18a) and (18b)
    Psi[0]     = np.sin(z1)       # Eq. (18a)
    Zeta[0]    = -ci * e**(ci * z1)
    PsiZeta[0] = 0.5 * (1 - e**(2 * ci * z1))
    D3[0] = ci
    for n in range(1, nmax): # Eq. (18b)
        Psi[n]  = Psi[<unsigned int>(n-1)]  * (n/z1 - D1[<unsigned int>(n-1)])
        Zeta[n] = Zeta[<unsigned int>(n-1)] * (n/z1 - D3[<unsigned int>(n-1)])
        PsiZeta[n] = PsiZeta[<unsigned int>(n-1)] * (n/z1 -D1[<unsigned int>(n-1)]) * (n/z1 - D3[<unsigned int>(n-1)])
        D3[n] = D1[n] + ci/PsiZeta[n]

    # Scattering coefficients, Eqs. (5) and (6)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] n2
    n2 = arange(nmax + 1, 0)

    # ab[0, :]  = (Ha[0, :]/m + n2/x) * Psi  - positive_roll(Psi,  1) # Eq. (5)
    # ab[0, :] /= (Ha[0, :]/m + n2/x) * Zeta - positive_roll(Zeta, 1)
    # ab[1, :]  = (Hb[0, :]*m + n2/x) * Psi  - positive_roll(Psi,  1) # Eq. (6)
    # ab[1, :] /= (Hb[0, :]*m + n2/x) * Zeta - positive_roll(Zeta, 1)

    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=1] roll_Psi
    cdef np.ndarray[DTYPE_COMPLEX_t, ndim=1] roll_Zeta

    roll_Psi = positive_roll(Psi,  1)
    roll_Zeta = positive_roll(Zeta,  1)

    for n in range(ab.shape[1]):
        ab[0, n]  = (((Ha[0, n]/m + n2[n]/x) * Psi[n]  - roll_Psi[n]) /
                     ((Ha[0, n]/m + n2[n]/x) * Zeta[n] - roll_Zeta[n])) # Eq. (5)
        ab[1, n]  = (((Hb[0, n]*m + n2[n]/x) * Psi[n]  - roll_Psi[n]) /
                     ((Hb[0, n]*m + n2[n]/x) * Zeta[n] - roll_Zeta[n])) # Eq. (6)

    for n in range(ab.shape[0]):
        ab[n, 0]  = 0 + 0*1j

    return ab
