from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

cimport numpy as np
DTYPE_COMPLEX = np.complex128
ctypedef np.complex128_t DTYPE_COMPLEX_t
DTYPE_FLOAT = np.float64
ctypedef np.float64_t DTYPE_FLOAT_t


def dcomplex(float a, float b):
    """
    Construct a complex number (or array) with the same call signature
    as the IDL ``dcomplex`` constructor.

    Parameters
    ----------
    a : float or `~numpy.ndarray`
        Real part of a complex number
    b : float or `~numpy.ndarray`
        Imaginary part of a complex number

    Returns
    -------
    A complex number or array of complex numbers.
    """
    cdef complex c = a + 1j*b
    return c

def dcomplexarr1d(int n):
    """
    Construct a complex array with the same call signature
    as the IDL ``dcomplexarr`` constructor.

    Parameters
    ----------
    n : int
        Length of complex array along first axis

    Returns
    -------
    A complex array.
    """
    cdef np.ndarray c = np.zeros(n, dtype=DTYPE_FLOAT) + 1j*np.zeros(n, dtype=DTYPE_FLOAT)
    return c


def dcomplexarr2d(int n, int m):
    """
    Construct a complex array with the same call signature
    as the IDL ``dcomplexarr`` constructor.

    Parameters
    ----------
    n : int
        Length of complex array along first axis
    m : int (optional)
        Length of complex array along second axis. If ``m`` is `None`, return
        only a one dimensional array

    Returns
    -------
    A complex array.
    """
    cdef np.ndarray c = np.zeros((n, m), dtype=DTYPE_FLOAT) + 1j*np.zeros((n, m), dtype=DTYPE_FLOAT)
    return c


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
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] x_i = np.arange(nx, dtype=DTYPE_FLOAT) - rp[0]
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] y_i = np.arange(ny, dtype=DTYPE_FLOAT) - rp[1]
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
    cdef int nc = np.size(ab[0,:])-1      # number of terms required for convergence

    cdef float k = 2 * np.pi / lambda_         # wavenumber in medium [pixel**-1]

    cdef complex ci = 0 + 1*1j

    # convert to spherical coordinates centered on the sphere.
    # (r, theta, phi) is the spherical coordinate of the pixel
    # at (x,y) in the imaging plane at distance z from the
    # center of the sphere.
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] rho = np.sqrt(x_**2 + y_**2)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] r = np.sqrt(rho**2 + z_**2)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] theta = np.arctan2(rho, z_)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] phi = np.arctan2(y_, x_)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] costheta = np.cos(theta).ravel()
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] sintheta = np.sin(theta).ravel()
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] cosphi = np.cos(phi).ravel()
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] sinphi = np.sin(phi).ravel()

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] kr = k*r.ravel()   # reduced radial coordinate

    # starting points for recursive function evaluation ...
    # ... Riccati-Bessel radial functions, page 478
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] sinkr = np.sin(kr)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] coskr = np.cos(kr)
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
    cdef int n
    for n in range(1, nc):

        # upward recurrences ...
        # ... Legendre factor (4.47)
        # Method described by Wiscombe (1980)
        swisc = pi_n * costheta
        twisc = swisc - pi_nm1
        tau_n = pi_nm1 - n * twisc  # -\tau_n(\cos\theta)

        # ... Riccati-Bessel function, page 478
        xi_n = (2*n - 1) * xi_nm1 / kr - xi_nm2    # \xi_n(kr)

        # vector spherical harmonics (4.50)
        Mo1n[1,:] = pi_n * xi_n     # ... divided by cosphi/kr
        Mo1n[2,:] = tau_n * xi_n    # ... divided by sinphi/kr

        dn = (n * xi_n)/kr - xi_nm1
        Ne1n[0,:] = n*(n + 1) * pi_n * xi_n # ... divided by cosphi sintheta/kr**2
        Ne1n[1,:] = tau_n * dn      # ... divided by cosphi/kr
        Ne1n[2,:] = pi_n  * dn      # ... divided by sinphi/kr

        # prefactor, page 93
        En = ci**n * (2*n + 1) / n / (n + 1)

        # the scattered field in spherical coordinates (4.45)
        Es += (En * ci * ab[0,n]) * Ne1n - (En * ab[1,n]) * Mo1n

        # upward recurrences ...
        # ... angular functions (4.47)
        # Method described by Wiscombe (1980)
        pi_nm1 = pi_n
        pi_n = swisc + ((n + 1) / n) * twisc

        # ... Riccati-Bessel function
        xi_nm2 = xi_nm1
        xi_nm1 = xi_n

    # geometric factors were divided out of the vector
    # spherical harmonics for accuracy and efficiency ...
    # ... put them back at the end.
    Es[0,:] *= cosphi * sintheta / kr**2
    Es[1,:] *= cosphi / kr
    Es[2,:] *= sinphi / kr
    # in IDL version by default the scattered wave is returned in spherical
    # coordinates.  Project components onto Cartesian coordinates.
    # Assumes that the incident wave propagates along z and
    # is linearly polarized along x

    # In python version, do cartesian always
    Ec = Es.copy()
    Ec[0, :] =  Es[0, :] * sintheta * cosphi
    Ec[0, :] += Es[1, :] * costheta * cosphi
    Ec[0, :] -= Es[2, :] * sinphi

    Ec[1, :] =  Es[0, :] * sintheta * sinphi
    Ec[1, :] += Es[1, :] * costheta * sinphi
    Ec[1, :] += Es[2, :] * cosphi

    Ec[2, :] =  Es[0, :] * costheta - Es[1, :] * sintheta

    return Ec

# def shift(np.ndarray a, int shift):
#     cdef np.ndarray r = np.roll(a, shift)
#     return r

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
    choice3 = np.abs(np.roll(x, -1)*m)
    max_choice = max([choice1, choice2, choice3])
    result = int(np.floor(max_choice) + 15)
    return result


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
    cdef unsigned int n
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

    # Calculate D1, D3 and PsiZeta for Z1 in the first layer
    #z1 = x[0] * m[0]
    cdef complex z1 = x * m
    for n in range(1, nmax+1)[::-1]:
        D1_a[0, n-1:n] = n/z1 - 1/(D1_a[0, n] + n/z1)

    PsiZeta_a[0, 0:1] = 0.5 * (1 - np.exp(2 * ci * z1)) # Eq. (18a)

    D3_a[0, 0] = ci                                     # Eq. (18a)
    for n in range(1, nmax):      #upward recurrence Eq. (18b)
        PsiZeta_a[n:n+1, 0] = (PsiZeta_a[0, n-1] *
                           (n/z1 - D1_a[0, n-1]) * (n/z1 - D3_a[0, n-1]))
        #D3_a[n:n+1, 0] = D1_a[0, n] + ci/PsiZeta_a[0, n]

    # Ha and Hb in the core
    Ha[0, :] = D1_a[0, :-1]     # Eq. (7a)
    Hb[0, :] = D1_a[0, :-1]     # Eq. (8a)


    # z1 = dcomplex(x[-1])
    z1 = x + 0*1j
    # Downward recurrence for D1, Eqs. (16a) and (16b)
    for n in range(1, nmax)[::-1]: # Eq. (16b)
        D1[n-1] = n/z1 - (1/(D1[n] + n/z1))

    # Upward recurrence for Psi, Zeta, PsiZeta and D3, Eqs. (18a) and (18b)
    Psi[0]     = np.sin(z1)       # Eq. (18a)
    Zeta[0]    = -ci * np.exp(ci * z1)
    PsiZeta[0] = 0.5 * (1 - np.exp(2 * ci * z1))
    D3[0] = ci
    for n in range(1, nmax): # Eq. (18b)
        Psi[n]  = Psi[n-1]  * (n/z1 - D1[n-1])
        Zeta[n] = Zeta[n-1] * (n/z1 - D3[n-1])
        PsiZeta[n] = PsiZeta[n-1] * (n/z1 -D1[n-1]) * (n/z1 - D3[n-1])
        D3[n] = D1[n] + ci/PsiZeta[n]

    # Scattering coefficients, Eqs. (5) and (6)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] n2 = np.arange(nmax + 1, dtype=DTYPE_FLOAT)

    ab[0, :]  = (Ha[-1, :]/m + n2/x) * Psi  - np.roll(Psi,  1) # Eq. (5)
    ab[0, :] /= (Ha[-1, :]/m + n2/x) * Zeta - np.roll(Zeta, 1)
    ab[1, :]  = (Hb[-1, :]*m + n2/x) * Psi  - np.roll(Psi,  1) # Eq. (6)
    ab[1, :] /= (Hb[-1, :]*m + n2/x) * Zeta - np.roll(Zeta, 1)
    ab[:, 0]  = 0 + 0*1j

    return ab
