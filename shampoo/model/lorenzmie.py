"""
Translation notes:

findgen is like np.arange(), except with one element it produces a row vector,
or with two, i.e. FINDGEN(1, 10), it will produce the same as np.arange(10).T
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import matplotlib.pyplot as plt

precision = None
delta = 0
alpha = 1

def lmsphere(rp, ap, n_sphere, nm, lambda_, mpp, dim, alpha = alpha, delta = delta,
             precision = precision):   # field = field,
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
    nx = float(dim[0])
    ny = float(dim[1])
    npts = nx * ny
    #x = np.arange(nx) - rp[0]
    #y = np.arange(1, ny).reshape((ny, 1)) - rp[1]
    #x = rebin(x, nx, ny, /sample)
    #y = rebin(y, nx, ny, /sample)
    x, y = np.meshgrid(np.arange(nx) - rp[0], np.arange(ny) - rp[1])

    zp = float(rp[2])

    field = spherefield(x, y, zp, ap, n_sphere, nm, lambda_, mpp)

    # if ~isa(alpha, /number, /scalar) then $
    #    alpha = 1
    #
    # if ~isa(delta, /number, /scalar) then $
    #    delta = 0

    # BM: Taken from spherefield
    lambda_m = lambda_ / np.real(nm) / mpp ## medium wavenp.sizegth [pixel]
    k = 2*np.pi / lambda_m

    print(field.shape)

    field *= alpha * np.exp(0. -k*(zp + delta)*1j) ## scattered field
    field[0, :] += 1.                                   ## incident field


    a = np.sum(np.real(field * np.conj(field)), 0)        ## intensity

    return a.reshape((nx, ny))

def spherefield(x, y, z, a, n_sphere, nm, lambda_, mpp):
    """
    x: [npts] array of pixel coordinates [pixels]
    y: [npts] array of pixel coordinates [pixels]
    z: If field is required in a single plane, then
       z is the plane's distance from the sphere's center.
       Otherwise, z is an [npts] array of coordinates.

       NOTE: Ideally, x, y and z should be double precision.
             This is left to the calling program for efficiency.

    ap: radius of sphere [micrometers]
    n_sphere: Complex refractive index of sphere
    nm: Complex refractive index of medium.
    lambda: vacuum wavenp.sizegth of light [micrometers]
    mpp: microns per pixel.
    """

    ab = sphere_coefficients(a, n_sphere, nm, lambda_)

    # if np.size(precision) eq 1 then begin
    #    fac = total(abs(ab[1,:]),1)
    #    w = where(fac gt precision*max(fac))
    #    ab = ab[*,[0,w]]             ## retain first coefficient for bookkeeping
    # endif

    lambda_m = lambda_ / nm.real / mpp ## medium wavenp.sizegth [pixel]

    k = 2*np.pi / lambda_m       ## wavenumber in medium [pixel**{-1}]

    field = sphericalfield(x, y, z, ab, lambda_m)# , cartesian=cartesian)

    return field

def dcomplex(a, b=0):
    return a + 1j*b

def dcomplexarr(n, m=None):
    if m is None:
        return np.zeros(n) + 1j*np.zeros(n)
    return np.zeros((n, m)) + 1j*np.zeros((n, m))

def sphericalfield(x_, y_, z_, ab, lambda_):
    """
    x: [npts] array of pixel coordinates [pixels]
    y: [npts] array of pixel coordinates [pixels]
    z: If field is required in a single plane, then z is the plane's
       distance from the sphere's center [pixels].
       Otherwise, z is an [npts] array of coordinates.

    NOTE: Ideally, x, y and z should be double precision.
          This is left to the calling program for efficiency.

    ab: [2,nc] array of Lorenz-Mie scattering coefficients, where
          nc is the number of terms required for convergence.

    lambda: wavenp.sizegth of light in medium [pixels]
    """
    npts = np.size(x_)
    # nc = np.size(ab[0,:])-1      # number of terms required for convergence
    nc = np.size(ab[0,:])-1      # number of terms required for convergence

    k = 2 * np.pi / lambda_         # wavenumber in medium [pixel**-1]

    ci = dcomplex(0, 1)

    # convert to spherical coordinates centered on the sphere.
    # (r, theta, phi) is the spherical coordinate of the pixel
    # at (x,y) in the imaging plane at distance z from the
    # center of the sphere.
    rho   = np.sqrt(x_**2 + y_**2)
    r     = np.sqrt(rho**2 + z_**2)
    theta = np.arctan2(rho, z_)
    phi   = np.arctan2(y_, x_)
    costheta = np.cos(theta).ravel()
    sintheta = np.sin(theta).ravel()
    cosphi = np.cos(phi).ravel()
    sinphi = np.sin(phi).ravel()

    kr = k*r.ravel()                        # reduced radial coordinate

    # starting points for recursive function evaluation ...
    # ... Riccati-Bessel radial functions, page 478
    sinkr = np.sin(kr)
    coskr = np.cos(kr)
    xi_nm2 = dcomplex(coskr, sinkr) # \xi_{-1}(kr)
    xi_nm1 = dcomplex(sinkr,-coskr) # \xi_0(kr)

    # ... angular functions (4.47), page 95
    pi_nm1 = 0                    # \pi_0(\cos\theta)
    pi_n   = 1                    # \pi_1(\cos\theta)

    # storage for vector spherical harmonics: [r,theta,phi]
    Mo1n = dcomplexarr(3, npts)
    Ne1n = dcomplexarr(3, npts)

    # storage for scattered field
    Es = dcomplexarr(3, npts)

    # Compute field by summing multipole contributions
    #for n = 1, nc do begin
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
        #   Mo1n[0,:] = 0             # no radial component
        # Mo1n[1,:] = pi_n * xi_n     # ... divided by cosphi/kr
        # Mo1n[2,:] = tau_n * xi_n    # ... divided by sinphi/kr
        #
        # dn = (n * xi_n)/kr - xi_nm1
        # Ne1n[0,:] = n*(n + 1) * pi_n * xi_n # ... divided by cosphi sintheta/kr**2
        # Ne1n[1,:] = tau_n * dn      # ... divided by cosphi/kr
        # Ne1n[2,:] = pi_n  * dn      # ... divided by sinphi/kr
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
    # By default, the scattered wave is returned in spherical
    # coordinates.  Project components onto Cartesian coordinates.
    # Assumes that the incident wave propagates along z and
    # is linearly polarized along x
    #if keyword_set(cartesian) then begin

    # BM: Do cartesian always
    Ec = Es.copy()
    Ec[0, :] =  Es[0, :] * sintheta * cosphi
    Ec[0, :] += Es[1, :] * costheta * cosphi
    Ec[0, :] -= Es[2, :] * sinphi

    Ec[1, :] =  Es[0, :] * sintheta * sinphi
    Ec[1, :] += Es[1, :] * costheta * sinphi
    Ec[1, :] += Es[2, :] * cosphi

    Ec[2, :] =  Es[0, :] * costheta - Es[1, :] * sintheta

    return Ec

def shift(a, shift):
    return np.roll(a, shift)

def Nstop(x, m):
    """
    Number of terms to keep in the partial wave expansion
    """
    #;; Wiscombe (1980)
    xl = x[-1]
    if xl < 8.:
        ns = np.floor(xl + 4. * xl**(1./3.) + 1.)
    elif xl < 4200.:
        ns = np.floor(xl + 4.05 * xl**(1./3.) + 2.)
    elif xl > 4199.:
        ns = np.floor(xl + 4. * xl**(1./3.) + 2.)

    #;; Yang (2003) Eq. (30)
    return int(np.floor(np.max([ns, abs(x*m), np.abs(shift(x, -1)*m)])) + 15)


def sphere_coefficients(ap, n_sphere, nm, lambda_):
    """
    ap: [nlayers] radii of layered sphere [micrometers]
        NOTE: ap and np are reordered automatically so that
        ap is in ascending order.

    n_sphere: [nlayers] (complex) refractive indexes of sphere's layers

    nm: (complex) refractive index of medium

    lambda: wavenp.sizegth of light [micrometers]
    """
    if not hasattr(ap, '__np.size__'):
        ap = np.array([ap])
    if not hasattr(n_sphere, '__np.size__'):
        n_sphere = np.array([n_sphere])

    nlayers = np.size(ap)

    # arrange shells in size order
    if nlayers > 1:
        order = np.argsort(ap)
        ap = ap[order]
        n_sphere = n_sphere[order]

    x = 2 * np.pi * np.real(nm) * ap / lambda_ # size parameter [array]
    m = dcomplex(n_sphere/nm)                    # relative refractive index [array]

    nmax = Nstop(x, m)              # number of terms in partial-wave expansion

    ci = dcomplex(0, 1)             # imaginary unit

    # arrays for storing results
    ab = dcomplexarr(2, nmax+1)#, /nozero)

    D1     = dcomplexarr(nmax+2)
    D1_a   = dcomplexarr(nlayers, nmax+2)
    D1_am1 = dcomplexarr(nlayers, nmax+2)

    D3     = dcomplexarr(nmax+1)
    D3_a   = dcomplexarr(nlayers, nmax+1)
    D3_am1 = dcomplexarr(nlayers, nmax+1)

    Psi         = dcomplexarr(nmax+1)
    Zeta        = dcomplexarr(nmax+1)
    PsiZeta     = dcomplexarr(nmax+1)
    PsiZeta_a   = dcomplexarr(nlayers, nmax+1)
    PsiZeta_am1 = dcomplexarr(nlayers, nmax+1)

    Q  = dcomplexarr(nlayers, nmax+1)
    Ha = dcomplexarr(nlayers, nmax+1)
    Hb = dcomplexarr(nlayers, nmax+1)

    # Calculate D1, D3 and PsiZeta for Z1 in the first layer
    z1 = x[0] * m#[0]
    # D1_a[0, nmax + 1] = dcomplex(0) # Eq. (16a)
    #for n = nmax + 1, 1, -1 do $ # downward recurrence Eq. (16b)
    for n in range(1, nmax+1)[::-1]:
        # D1_a[0, n-1] = n/z1 - 1/(D1_a[0, n] + n/z1)
        D1_a[0, n-1:n] = n/z1 - 1/(D1_a[0, n] + n/z1)

    # PsiZeta_a[0, 0] = 0.5 * (1 - np.exp(2 * ci * z1)) # Eq. (18a)
    PsiZeta_a[0, 0:1] = 0.5 * (1 - np.exp(2 * ci * z1)) # Eq. (18a)

    D3_a[0, 0] = ci                                     # Eq. (18a)
    #for n = 1, nmax do begin      #upward recurrence Eq. (18b)
    for n in range(1, nmax):      #upward recurrence Eq. (18b)
        # PsiZeta_a[0, n] = (PsiZeta_a[0, n-1] *
        #                    (n/z1 - D1_a[0, n-1]) * (n/z1 - D3_a[0, n-1]))
        # D3_a[0, n] = D1_a[0, n] + ci/PsiZeta_a[0, n]
        PsiZeta_a[n:n+1, 0] = (PsiZeta_a[0, n-1] *
                           (n/z1 - D1_a[0, n-1]) * (n/z1 - D3_a[0, n-1]))
        D3_a[n:n+1, 0] = D1_a[0, n] + ci/PsiZeta_a[0, n]

    # Ha and Hb in the core
    # Ha[0, :] = D1_a[0, 0:-2]     # Eq. (7a)
    # Hb[0, :] = D1_a[0, 0:-2]     # Eq. (8a)
    Ha[0, :] = D1_a[0, :-1]     # Eq. (7a)
    Hb[0, :] = D1_a[0, :-1]     # Eq. (8a)

    # Iterate from layer 2 to layer L
    #for ii = 1, nlayers - 1 do begin
    for ii in range(1, nlayers - 1):
        z1 = x[ii] * m[ii]
        z2 = x[ii-1] * m[ii]
        # Downward recurrence for D1, Eqs. (16a) and (16b)
        #   D1_a[ii, nmax+1]   = dcomplex(0)      # Eq. (16a)
        #   D1_am1[ii, nmax+1] = dcomplex(0)
        #for n = nmax + 1, 1, -1 do begin # Eq. (16 b)
        for n in range(1, nmax + 1)[::-1]: # Eq. (16 b)
            D1_a[n-1,ii]   = n/z1 - 1/(D1_a[ii, n]   + n/z1)
            D1_am1[n-1,ii] = n/z2 - 1/(D1_am1[ii, n] + n/z2)


        # Upward recurrence for PsiZeta and D3, Eqs. (18a) and (18b)
        PsiZeta_a[0, ii]   = 0.5 * (1 - np.exp(2 * ci * z1)) # Eq. (18a)
        PsiZeta_am1[0, ii] = 0.5 * (1 - np.exp(2 * ci * z2))
        D3_a[0, ii]   = ci
        D3_am1[0, ii] = ci
        #for n = 1, nmax do begin   # Eq. (18b)
        for n in range(1, nmax):   # Eq. (18b)
            PsiZeta_a[n, ii]   = (PsiZeta_a[n-1, ii] *
                                  (n/z1 -  D1_a[n-1, ii]) *
                                  (n/z1 -  D3_a[n-1, ii]))
            PsiZeta_am1[n, ii] = (PsiZeta_am1[n-1, ii] *
                                  (n/z2 - D1_am1[n-1, ii]) *
                                  (n/z2 - D3_am1[n-1, ii]))
            D3_a[n, ii]   = D1_a[n, ii]   + ci/PsiZeta_a[n, ii]
            D3_am1[n, ii] = D1_am1[n, ii] + ci/PsiZeta_am1[n, ii]


        # Upward recurrence for Q
        Q[ii, 0] = (np.exp(-2 * ci * z2) - 1) / (np.exp(-2 * ci * z1) - 1)
        for n in range(1, nmax):
            Num = (z1 * D1_a[n,ii]   + n) * (n - z1 * D3_a[n-1, ii])
            Den = (z2 * D1_am1[n,ii] + n) * (n - z2 * D3_am1[n-1, ii])
            Q[n,ii] = (x[ii-1]/x[ii])**2 * Q[n-1, ii] * Num/Den


        # Upward recurrence for Ha and Hb, Eqs. (7b), (8b) and (12) - (15)
        for n in range(1, nmax):
            G1 = m[ii] * Ha[n, ii-1] - m[ii-1] * D1_am1[n, ii]
            G2 = m[ii] * Ha[n, ii-1] - m[ii-1] * D3_am1[n, ii]
            Temp = Q[n, ii] * G1
            Num = G2 * D1_a[n, ii] - Temp * D3_a[n, ii]
            Den = G2 - Temp
            Ha[n, ii] = Num/Den

            G1 = m[ii-1] * Hb[n, ii-1] - m[ii] * D1_am1[n, ii]
            G2 = m[ii-1] * Hb[n, ii-1] - m[ii] * D3_am1[n, ii]
            Temp = Q[n, ii] * G1
            Num = G2 * D1_a[n, ii] - Temp * D3_a[n, ii]
            Den = G2 - Temp
            Hb[n, ii] = Num/Den

            #ii (layers)

    z1 = dcomplex(x[-1])
    # Downward recurrence for D1, Eqs. (16a) and (16b)
    # D1[nmax+1] = dcomplex(0)   # Eq. (16a)
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
    n = np.arange(nmax + 1)
    # ab[0, :]  = (Ha[-1, :]/m[-1] + n/x[-1]) * Psi  - shift(Psi,  1) # Eq. (5)
    # ab[0, :] /= (Ha[-1, :]/m[-1] + n/x[-1]) * Zeta - shift(Zeta, 1)
    # ab[1, :]  = (Hb[-1, :]*m[-1] + n/x[-1]) * Psi  - shift(Psi,  1) # Eq. (6)
    # ab[1, :] /= (Hb[-1, :]*m[-1] + n/x[-1]) * Zeta - shift(Zeta, 1)
    # ab[0, :]  = (((Ha[:, -1][:, np.newaxis]/m[-1] + n/x[-1]) * Psi  - shift(Psi,  1)) /
    #              (Ha[:, -1][:, np.newaxis]/m[-1] + n/x[-1]) * Zeta - shift(Zeta, 1)) # Eq. (5)
    # #ab[0, :] /= (Ha[:, -1][:, np.newaxis]/m[-1] + n/x[-1]) * Zeta - shift(Zeta, 1)
    # ab[1, :]  = (((Hb[:, -1][:, np.newaxis]*m[-1] + n/x[-1]) * Psi  - shift(Psi,  1)) /
    #              (Hb[:, -1][:, np.newaxis]*m[-1] + n/x[-1]) * Zeta - shift(Zeta, 1)) # Eq. (6)
    # #ab[1, :] /= (Hb[:, -1][:, np.newaxis]*m[-1] + n/x[-1]) * Zeta - shift(Zeta, 1)
    # ab[:, 0]  = dcomplex(0)

    # ab[:, 0]  = (Ha[-1, :]/m[-1] + n/x[-1]) * Psi  - shift(Psi,  1) # Eq. (5)
    # ab[:, 0] /= (Ha[-1, :]/m[-1] + n/x[-1]) * Zeta - shift(Zeta, 1)
    # ab[:, 1]  = (Hb[-1, :]*m[-1] + n/x[-1]) * Psi  - shift(Psi,  1) # Eq. (6)
    # ab[:, 1] /= (Hb[-1, :]*m[-1] + n/x[-1]) * Zeta - shift(Zeta, 1)

    ab[0, :]  = (Ha[-1, :]/m[-1] + n/x[-1]) * Psi  - shift(Psi,  1) # Eq. (5)
    ab[0, :] /= (Ha[-1, :]/m[-1] + n/x[-1]) * Zeta - shift(Zeta, 1)
    ab[1, :]  = (Hb[-1, :]*m[-1] + n/x[-1]) * Psi  - shift(Psi,  1) # Eq. (6)
    ab[1, :] /= (Hb[-1, :]*m[-1] + n/x[-1]) * Zeta - shift(Zeta, 1)
    ab[:, 0]  = dcomplex(0)

    # if keyword_set(resolution) then begin
    #    w = where(total(abs(ab), 1) gt resolution, ngood)
    #    if ngood ge 1 then $
    #      ab = ab[*,0:w[-1]]
    # endif

    return ab


holo = lmsphere([0, 0, 200], 0.75, 1.5, 1.33, 0.532, 0.135, [201, 201])
#print(holo.shape)
plt.imshow(holo, cmap=plt.cm.binary, origin='lower')
plt.show()