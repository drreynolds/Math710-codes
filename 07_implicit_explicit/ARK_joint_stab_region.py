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
          S(theta) = { zI = -a+i*b : a>0, b>=0, and atan(b/a) <= theta }

        We note that the sector S(theta) contains infinitely many points:
        (a) it extends arbitrarily far into the complex left half-plane,
            i.e., |zI| < infty, and
        (b) it contains infinitely many angles 0 <= alpha <= theta.

        However, we only test this for the two angles alpha=0 and
        alpha=theta, using NI points, with distance logarithmically-
        scaled away from the origin, to a maximum distance of rmax.
        Similarly, we only test a NE^2 mesh of points zE within the
        pre-defined "box" in the complex plane.

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
                     assume that yl=-yr, and that the joint stability region
                     is symmetric across the real axis.
           fig    -- matplotlib figure handle to use
    '''

    # set general parameters
    NE = 101   # must be odd
    NI = 100
    Rthresh = 1.25
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

    # create e, I
    e = np.ones(s)
    I = np.eye(s)

    # construct the ARK stability function
    R = lambda zE, zI: 1 + np.dot(zE*bE + zI*bI, np.linalg.solve(I - zE*AE - zI*AI, e))

    # set mesh of ERK sample points
    xl = box[0]
    xr = box[1]
    yl = box[2]
    yr = box[3]
    x = np.linspace(xl, xr, NE)
    y = np.linspace(yl, yr, NE)

    # create new figure window
    ax = fig.gca()
    ax.plot(np.linspace(xl, xr, 10), np.zeros(10), 'k:')
    ax.plot(np.zeros(10), np.linspace(yl, yr, 10), 'k:')

    # loop over theta values, creating contour plot data for each
    mid = (NE-1)//2
    handles = []
    labels = []
    for itheta in range(len(thetas)):
        theta = thetas[itheta]*np.pi/180   # convert to radians

        # initialize max|R| over box
        Rmax = np.zeros((NE, NE))

        # set array of DIRK sample points
        r = -np.logspace(-1, np.log10(rmax), NI)
        zI = np.concatenate(([0], r, r*(np.cos(theta)-np.sin(theta)*1j)))

        # loop over zE mesh
        for j in range(0, mid+1):
            j1 = mid+j
            j2 = mid-j
            for i in range(NE):

                # set zE value
                zE = x[i] + y[mid+j]*1j

                # loop over zI values, breaking the moment |R(zE,zI)| > Rthresh
                for k in range(len(zI)):
                    Rval = abs(R(zE, zI[k]))
                    Rmax[j1, i] = max(Rmax[j1, i], Rval)
                    Rmax[j2, i] = max(Rmax[j2, i], Rval)
                    if (Rval > Rthresh):
                        break

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
