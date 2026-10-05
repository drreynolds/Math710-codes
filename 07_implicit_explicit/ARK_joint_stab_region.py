#!/usr/bin/env python
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

# general imports
import numpy as np
import matplotlib.pyplot as plt

def ARK_joint_stab_region(BE, BI, thetas, rmax, box, fig):
    ''' Usage: ARK_joint_stab_region(BE, BI, thetas, rmax, box, fig)

        Computes the joint stability region for an ARK method applied
        to the scalar, additive test problem,

           y'(t) = lI*y + lE*y,

        using time step h, where lI,lE in C, and Re(lI) < 0, Re(lE) < 0.
        For an s-stage ARK method applied to this problem, then defining
        the step-size scaled inputs zI = h*lI, zE = h*lE, the function
        is given by

          R(zE,zI) = 1 + (zE*bE + zI*bI)*((I-zE*AE-zI*AI)^{-1} e)

        where e is a vector of all ones in R^s.

        For a given angle 0 <= theta <= pi/2, we define the joint stability
        region as

          Sj(theta) = { zE in C : |R(zE,zI)|<1 forall zI in S(theta) }, where
          S(theta) = { zI = -a+i*b : a>0, and atan(|b|/a) <= theta }

        We note that the sector S(theta) contains infinitely many points:
        (a) it extends arbitrarily far into the complex left half-plane,
            i.e., |zI| < infty, and
        (b) it contains infinitely many angles -theta <= alpha <= theta.

        However, we need not test all of them.  Since the implicit table is
        diagonally implicit, the poles of R in zI are at 1/AI[i,i] > 0,
        outside of S(theta), and so for fixed zE the maximum of |R| over
        S(theta) is attained on its boundary (the maximum modulus principle),
        i.e., on the two rays alpha = -theta and alpha = theta.  Moreover,
        since the Butcher tables are real, R(conj(zE),conj(zI)) =
        conj(R(zE,zI)).  We therefore test only zI = 0 and the ray
        alpha = theta, using NI points, with distance logarithmically-
        scaled away from the origin, to a maximum distance of rmax, over the
        full NE^2 mesh of points zE within the pre-defined "box" in the
        complex plane, and then account for the ray alpha = -theta by
        reflecting the result across the real axis.  Finally, since
        I - zE*AE - zI*AI is lower triangular, we evaluate R by forward
        substitution over the whole zE mesh at once for each zI sample.

        The input 'thetas' is array-valued -- we plot the joint stability
        region Sj(theta) for each value in this array, and overlay these plots.

        Inputs:
           BE     -- ERK Butcher table dictionary, with fields 'A', 'b'
           BI     -- DIRK Butcher table dictionary, with fields 'A', 'b'
           thetas -- array of sector angles (in degrees) to use in creating
                     overlaid plots
           rmax   -- maximum distance from the origin for the DIRK sample points
           box    -- [xl, xr, yl, yr] is the bounding box for the sub-region
                     of the complex plane in which to perform the test.  We
                     assume that yl=-yr, so that the mesh is symmetric
                     across the real axis.
           fig    -- matplotlib figure handle to use
    '''

    # set general parameters
    NE = 101   # must be odd
    NI = 100
    CM = plt.get_cmap('tab10')

    # check Butcher tables for compatibility
    AE = np.asarray(BE['A'], dtype=float)
    bE = np.asarray(BE['b'], dtype=float)
    AI = np.asarray(BI['A'], dtype=float)
    bI = np.asarray(BI['b'], dtype=float)
    s = len(bE)
    if (len(bE) != len(bI)):
        raise ValueError('ARK_joint_stab_region: bE and bI are incompatible')
    if (AE.shape != (s, s)):
        raise ValueError('ARK_joint_stab_region: incompatible explicit Butcher table inputs')
    if (AI.shape != (s, s)):
        raise ValueError('ARK_joint_stab_region: incompatible implicit Butcher table inputs')
    if ((np.linalg.norm(AE - np.tril(AE,-1), np.inf) > 1e-14) or
        (np.linalg.norm(AI - np.tril(AI), np.inf) > 1e-14)):
        raise ValueError('ARK_joint_stab_region: requires an explicit and a diagonally implicit table')

    # set mesh of ERK sample points
    xl = box[0]
    xr = box[1]
    yl = box[2]
    yr = box[3]
    x = np.linspace(xl, xr, NE)
    y = np.linspace(yl, yr, NE)
    X, Y = np.meshgrid(x, y)
    ZE = X + Y*1j

    # construct the ARK stability function over the whole zE mesh, for a
    # single zI value: since I - zE*AE - zI*AI is lower triangular, the stages
    #    z_i = (1 + sum_{j<i} (zE*AE[i,j] + zI*AI[i,j]) z_j) / (1 - zI*AI[i,i])
    # follow by forward substitution, and R = 1 + sum_i (zE*bE[i] + zI*bI[i]) z_i
    def R(ZE, zI):
        z = []
        for i in range(s):
            zi = np.ones_like(ZE)
            for j in range(i):
                zi = zi + (ZE*AE[i,j] + zI*AI[i,j])*z[j]
            z.append(zi/(1 - zI*AI[i,i]))
        return 1 + sum((ZE*bE[i] + zI*bI[i])*z[i] for i in range(s))

    # create new figure window
    ax = fig.gca()
    ax.plot(np.linspace(xl, xr, 10), np.zeros(10), 'k:')
    ax.plot(np.zeros(10), np.linspace(yl, yr, 10), 'k:')

    # loop over theta values, creating contour plot data for each
    handles = []
    labels = []
    for itheta in range(len(thetas)):
        theta = thetas[itheta]*np.pi/180   # convert to radians

        # initialize max|R| over box
        Rmax = np.zeros((NE, NE))

        # set array of DIRK sample points: zI = 0, and the upper ray of S(theta)
        r = -np.logspace(-1, np.log10(rmax), NI)
        zI = np.concatenate(([0], r*(np.cos(theta)-np.sin(theta)*1j)))

        # compute max|R| over the zI samples, for the whole zE mesh at once,
        # and then reflect across the real axis to account for the lower ray
        for k in range(len(zI)):
            Rmax = np.maximum(Rmax, np.abs(R(ZE, zI[k])))
        Rmax = np.maximum(Rmax, Rmax[::-1,:])

        # create contour and add to figure
        eps = np.finfo(float).eps
        c = ax.contour(x, y, Rmax, levels=[1+eps], colors=[CM(itheta)], linewidths=2)

        # assemble plot
        lstring = r'$\theta =\;$' + '%g' % (thetas[itheta])
        handle, _ = c.legend_elements()
        handles.append(handle[0])
        labels.append(lstring)

    # finish up figure
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_xlabel('Re(zE)')
    ax.set_ylabel('Im(zE)')
    ax.legend(handles, labels, loc='upper left')


# end of file
