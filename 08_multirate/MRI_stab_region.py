#!/usr/bin/env python
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

# general imports
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def phi_functions(z, n):
    ''' Usage: phi = phi_functions(z, n)

        Returns the array [phi_0(z), phi_1(z), ..., phi_n(z)], where

           phi_0(z) = e^z,
           phi_k(z) = 1/(k-1)! int_0^1 e^{(1-theta)z} theta^{k-1} dtheta,  k >= 1.

        These are computed together from the exponential of the
        (n+1)x(n+1) matrix

           [ z 1 0 ... 0 ]
           [ 0 0 1 ... 0 ]
           [ ...     ... ]
           [ 0 0 0 ... 1 ]
           [ 0 0 0 ... 0 ],

        whose first row is [phi_0(z), ..., phi_n(z)].  Unlike the recurrence
        phi_{k+1}(z) = (phi_k(z) - 1/k!)/z, this is accurate for small |z|.
    '''
    A = np.diag(np.ones(n, dtype=complex), 1)
    A[0,0] = z
    return expm(A)[0,:]

def MRI_stab_region(C, alphas, rho, box, ax):
    r''' Usage: MRI_stab_region(C, alphas, rho, box, ax)

        Computes the slow stability region of an explicit MRI-GARK method
        applied to the scalar, additive test problem,

           y'(t) = lF*y + lS*y,

        using slow time step H, where lF,lS \in \C, and Re(lF) < 0, Re(lS) < 0.
        Defining zF = H*lF and zS = H*lS, each MRI stage solves a linear fast
        IVP with polynomial forcing exactly, and so (see the lecture notes)

           z_1 = 1,
           z_i = phi_0(dc_i zF) z_{i-1}
                 + zS sum_{j<i} ( sum_k k! G[k,i,j] phi_{k+1}(dc_i zF) ) z_j,

        for i = 2,...,s, where dc_i = c_i - c_{i-1}, and R(zF,zS) = z_s.

        For a radius rho > 0 (possibly np.inf) and an angle 0 <= alpha < 90
        degrees, the slow stability region is

           S_{rho,alpha} = { zS \in \C : |R(zF,zS)| <= 1 for all zF in W }, where
           W = { zF \in \C : |zF| <= rho, |arg(zF) - pi| <= alpha }.

        We use two facts to avoid sampling all of the wedge W:
        (a) For fixed zS, R is an analytic and bounded function of zF on W,
            so the maximum of |R| over W is attained on its boundary (the
            maximum modulus principle; for rho = inf, its extension to
            sectors, the Phragmen--Lindelof principle).  We therefore only
            sample the two rays arg(zF) = pi -/+ alpha out to |zF| = rho, and
            the arc |zF| = rho between them.
        (b) Since the MRI coefficients are real, R(conj(zF),conj(zS)) =
            conj(R(zF,zS)).  We therefore sample only the upper half of the
            boundary of W (one ray and half of the arc) over the full zS
            mesh, and then account for the lower half by reflecting the
            result across the real axis.
        For each fast sample, we evaluate R over the whole zS mesh at once.

        The input 'alphas' is array-valued -- we plot the slow stability
        region S_{rho,alpha} for each value in this array, and overlay these
        plots on the stability region of the slow base method (zF = 0).

        Inputs:
           C      -- explicit MRI coupling table dictionary, with fields 'G'
                     and 'c' (as in MRI.py)
           alphas -- array of wedge angles (in degrees, < 90) to use in
                     creating overlaid plots
           rho    -- radius of the wedge W (np.inf is allowed)
           box    -- [xl, xr, yl, yr] is the bounding box for the sub-region
                     of the complex plane in which to perform the test.  We
                     assume that yl=-yr, so that the stability region is
                     symmetric across the real axis.
           ax     -- matplotlib axes handle to use
    '''

    # set general parameters
    NS = 401   # zS mesh points in each direction (must be odd)
    NR = 400   # zF samples along the ray
    NA = 50    # zF samples along the half-arc (for rho < inf)
    CM = plt.get_cmap('tab10')

    # extract MRI coefficients, and check that the method is explicit
    G = np.asarray(C['G'], dtype=float)
    c = np.asarray(C['c'], dtype=float)
    K = np.size(G, 0)
    s = c.size
    dc = np.diff(c)
    for k in range(K):
        if (np.linalg.norm(G[k] - np.tril(G[k], -1), np.inf) > 1e-14):
            raise ValueError('MRI_stab_region: only explicit MRI methods are supported')

    # set mesh of zS sample points
    xl = box[0]
    xr = box[1]
    yl = box[2]
    yr = box[3]
    x = np.linspace(xl, xr, NS)
    y = np.linspace(yl, yr, NS)
    X, Y = np.meshgrid(x, y)
    zS = X + Y*1j

    # utility routine to evaluate |R(zF,zS)| over the whole zS mesh
    def absR(zF):
        z = [np.ones_like(zS)]
        for i in range(1, s):
            phi = phi_functions(dc[i-1]*zF, K)
            znew = phi[0]*z[i-1]
            for j in range(i):
                gamma = sum(math.factorial(k)*G[k,i,j]*phi[k+1] for k in range(K))
                znew = znew + zS*gamma*z[j]
            z.append(znew)
        return np.abs(z[s-1])

    # create axes and plot the base method stability region (zF = 0)
    eps = np.finfo(float).eps
    ax.plot(np.linspace(xl, xr, 10), np.zeros(10), 'k:')
    ax.plot(np.zeros(10), np.linspace(yl, yr, 10), 'k:')
    cs = ax.contour(x, y, absR(0.0), levels=[1+eps], colors='k', linewidths=2)
    handle, _ = cs.legend_elements()
    handles = [handle[0]]
    labels = ['Base']

    # loop over alpha values, creating contour plot data for each
    for ialpha in range(len(alphas)):
        alpha = alphas[ialpha]*np.pi/180   # convert to radians

        # set array of zF sample points on the upper half of the boundary of W;
        # for rho = inf we sample the ray to |zF| = 1e6, where |R| has reached
        # its limiting value
        if (np.isinf(rho)):
            r = np.logspace(-3, 6, NR)
            zF = r*np.exp((np.pi - alpha)*1j)
        else:
            r = np.logspace(-3, np.log10(rho), NR)
            beta = np.linspace(np.pi - alpha, np.pi, NA)
            zF = np.concatenate((r*np.exp((np.pi - alpha)*1j), rho*np.exp(beta*1j)))

        # compute max|R| over these zF samples, and reflect across the real axis
        Rmax = absR(0.0)
        for k in range(zF.size):
            Rmax = np.maximum(Rmax, absR(zF[k]))
        Rmax = np.maximum(Rmax, Rmax[::-1,:])

        # create contour and add to figure
        cs = ax.contour(x, y, Rmax, levels=[1+eps], colors=[CM(ialpha)], linewidths=2)

        # assemble plot
        lstring = r'$\alpha =\;$' + '%g' % (alphas[ialpha]) + r'$^\circ$'
        handle, _ = cs.legend_elements()
        handles.append(handle[0])
        labels.append(lstring)

    # finish up figure
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect('equal')
    ax.set_xlabel('Re(zS)')
    ax.set_ylabel('Im(zS)')
    ax.legend(handles, labels, loc='upper left')


# end of file
