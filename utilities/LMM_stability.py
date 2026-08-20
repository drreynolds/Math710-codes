#!/usr/bin/env python3
# Function to generate and plot the linear stability regions for linear multistep
# methods.  Includes a "main" that uses this function to plot overlaid stability
# regions for Adams-Bashforth, Adams-Moulton, and Backwards Differentiation Formulas.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

# general imports
import numpy as np
import matplotlib.pyplot as plt

# routine to evaluate the LMM stability boundary function, eta(theta)
def LMM_stability(thetas, a, b):
    """
    Usage: x, y = LMM_stability(thetas,a,b):

    This function evaluates the boundary of the stability region for the LMM
    defined by the arrays a and b, for a given set of input angles, thetas.

    Inputs:   thetas - angles in the complex plane
              a, b - LMM arrays alpha_i and beta_i
    Outputs:  x, y - coordinate locations in the complex plane: x+i*y
    """

    # allocate outputs
    x = np.zeros(len(thetas))
    y = np.zeros(len(thetas))

    # compute the number of steps for this LMM
    n = len(a)-1
    m = len(b)-1
    k = max(n,m)

    # loop over thetas, filling output arrays
    for ith in range(len(thetas)):

        # compute xi
        xi = np.exp(complex(0.0,thetas[ith]))

        # compute the stability region numerator
        num = 0.0
        for j in range(n+1):
            num += a[j]*xi**(k-j)

        # compute the stability region denominator
        den = 0.0
        for j in range(m+1):
            den += b[j]*xi**(k-j)

        # compute the stability region boundary
        eta = num/den

        # extract the real and imaginary parts
        x[ith] = eta.real
        y[ith] = eta.imag

    return [x,y]


if __name__ == '__main__':
    ''' Driver that calls LMM_stability to plot overlaid stability regions for
        Adams-Bashforth, Adams-Moulton, and Backwards Differentiation Formulas.'''

    # set LMM coefficients for each method
    #   Adams-Bashforth
    AB1_a = np.array([1, -1])
    AB1_b = np.array([0, 1])
    AB2_a = np.array([1, -1])
    AB2_b = np.array([0, 3,-1])/2
    AB3_a = np.array([1, -1])
    AB3_b = np.array([0, 23,-16,5])/12
    AB4_a = np.array([1, -1])
    AB4_b = np.array([0, 55,-59,37,-9])/24
    AB5_a = np.array([1, -1])
    AB5_b = np.array([0, 1901, -2774, 2616, -1274, 251])/720

    #   Adams-Moulton
    AM1_a = np.array([1, -1])
    AM1_b = np.array([1])
    AM2_a = np.array([1, -1])
    AM2_b = np.array([1, 1])/2
    AM3_a = np.array([1, -1])
    AM3_b = np.array([5, 8, -1])/12
    AM4_a = np.array([1, -1])
    AM4_b = np.array([9, 19, -5, 1])/24
    AM5_a = np.array([1, -1])
    AM5_b = np.array([251, 646, -264, 106, -19])/720
    AM6_a = np.array([1, -1])
    AM6_b = np.array([475, 1427, -798, 482, -173, 27])/1440

    #   BDF
    BDF1_a = np.array([1, -1])
    BDF1_b = np.array([1])
    BDF2_a = np.array([1, -4/3, 1/3])
    BDF2_b = np.array([2/3])
    BDF3_a = np.array([1, -18/11, 9/11, -2/11])
    BDF3_b = np.array([6/11])
    BDF4_a = np.array([1, -48/25, 36/25, -16/25, 3/25])
    BDF4_b = np.array([12/25])
    BDF5_a = np.array([1, -300/137, 300/137, -200/137, 75/137, -12/137])
    BDF5_b = np.array([60/137])
    BDF6_a = np.array([1, -360/147, 450/147, -400/147, 225/147, -72/147, 10/147])
    BDF6_b = np.array([60/147])


    # set the thetas resolution
    nthetas = 1000
    thetas = np.linspace(0, 2 * np.pi, nthetas)

    # set the bounding box for plots in the complex plane
    box = [-6, 2, -4, 4]
    zoombox = [-1, 1, -3, 3]

    # set the transparency value for filled regions
    alp = 0.2


    # plot the Adams-Bashforth stability regions, one at a time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = LMM_stability(thetas, AB1_a, AB1_b)
    plt.fill(x, y, "b", alpha=alp)
    plt.plot(x, y, "b", label='$p=1$')
    x, y = LMM_stability(thetas, AB2_a, AB2_b)
    plt.fill(x, y, "r", alpha=alp)
    plt.plot(x, y, "r", label='$p=2$')
    x, y = LMM_stability(thetas, AB3_a, AB3_b)
    plt.fill(x, y, "k", alpha=alp)
    plt.plot(x, y, "k", label='$p=3$')
    x, y = LMM_stability(thetas, AB4_a, AB4_b)
    plt.fill(x, y, "g", alpha=alp)
    plt.plot(x, y, "g", label='$p=4$')
    x, y = LMM_stability(thetas, AB5_a, AB5_b)
    plt.fill(x, y, "m", alpha=alp)
    plt.plot(x, y, "m", label='$p=5$')
    plt.legend()
    plt.grid(True)
    plt.plot([box[0],box[1]],[0,0],'k--')
    plt.plot([0,0],[box[2],box[3]],'k--')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(box[0],box[1])
    ax.set_ylim(box[2],box[3])
    plt.title('Adams-Bashforth Stability Regions (shaded = stable)')
    plt.savefig('AB_stability.pdf')


    # plot the Adams-Moulton stability regions, one at a time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = LMM_stability(thetas, AM1_a, AM1_b)
    plt.fill(x, y, "b", alpha=alp)
    plt.plot(x, y, "b", label='$p=1$')
    x, y = LMM_stability(thetas, AM2_a, AM2_b)
    plt.fill(x, y, "r", alpha=alp)
    plt.plot(x, y, "r", label='$p=2$')
    x, y = LMM_stability(thetas, AM3_a, AM3_b)
    plt.fill(x, y, "k", alpha=alp)
    plt.plot(x, y, "k", label='$p=3$')
    x, y = LMM_stability(thetas, AM4_a, AM4_b)
    plt.fill(x, y, "g", alpha=alp)
    plt.plot(x, y, "g", label='$p=4$')
    x, y = LMM_stability(thetas, AM5_a, AM5_b)
    plt.fill(x, y, "m", alpha=alp)
    plt.plot(x, y, "m", label='$p=5$')
    x, y = LMM_stability(thetas, AM6_a, AM6_b)
    plt.fill(x, y, "c", alpha=alp)
    plt.plot(x, y, "c", label='$p=6$')
    plt.legend()
    plt.grid(True)
    plt.plot([box[0],box[1]],[0,0],'k--')
    plt.plot([0,0],[box[2],box[3]],'k--')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(box[0],box[1])
    ax.set_ylim(box[2],box[3])
    plt.title('Adams-Moulton Stability Regions (shaded = stable)')
    plt.savefig('AM_stability.pdf')


    # plot the BDF stability regions, one at a time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = LMM_stability(thetas, BDF1_a, BDF1_b)
    plt.fill(x, y, "b", alpha=alp)
    plt.plot(x, y, "b", label='$p=1$')
    x, y = LMM_stability(thetas, BDF2_a, BDF2_b)
    plt.fill(x, y, "r", alpha=alp)
    plt.plot(x, y, "r", label='$p=2$')
    x, y = LMM_stability(thetas, BDF3_a, BDF3_b)
    plt.fill(x, y, "k", alpha=alp)
    plt.plot(x, y, "k", label='$p=3$')
    x, y = LMM_stability(thetas, BDF4_a, BDF4_b)
    plt.fill(x, y, "g", alpha=alp)
    plt.plot(x, y, "g", label='$p=4$')
    x, y = LMM_stability(thetas, BDF5_a, BDF5_b)
    plt.fill(x, y, "m", alpha=alp)
    plt.plot(x, y, "m", label='$p=5$')
    x, y = LMM_stability(thetas, BDF6_a, BDF6_b)
    plt.fill(x, y, "c", alpha=alp)
    plt.plot(x, y, "c", label='$p=6$')
    plt.legend()
    plt.grid(True)
    plt.plot([box[0],box[1]],[0,0],'k--')
    plt.plot([0,0],[box[2],box[3]],'k--')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(box[0],box[1])
    ax.set_ylim(box[2],box[3])
    plt.title('BDF Stability Regions (shaded = unstable)')
    plt.savefig('BDF_stability.pdf')


    # plot a zoomed-in version of BDF stability regions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = LMM_stability(thetas, BDF1_a, BDF1_b)
    plt.fill(x, y, "b", alpha=alp)
    plt.plot(x, y, "b", label='$p=1$')
    x, y = LMM_stability(thetas, BDF2_a, BDF2_b)
    plt.fill(x, y, "r", alpha=alp)
    plt.plot(x, y, "r", label='$p=2$')
    x, y = LMM_stability(thetas, BDF3_a, BDF3_b)
    plt.fill(x, y, "k", alpha=alp)
    plt.plot(x, y, "k", label='$p=3$')
    x, y = LMM_stability(thetas, BDF4_a, BDF4_b)
    plt.fill(x, y, "g", alpha=alp)
    plt.plot(x, y, "g", label='$p=4$')
    x, y = LMM_stability(thetas, BDF5_a, BDF5_b)
    plt.fill(x, y, "m", alpha=alp)
    plt.plot(x, y, "m", label='$p=5$')
    x, y = LMM_stability(thetas, BDF6_a, BDF6_b)
    plt.fill(x, y, "c", alpha=alp)
    plt.plot(x, y, "c", label='$p=6$')
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 0.9))
    plt.grid(True)
    plt.plot([zoombox[0],zoombox[1]],[0,0],'k--')
    plt.plot([0,0],[zoombox[2],zoombox[3]],'k--')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(zoombox[0],zoombox[1])
    ax.set_ylim(zoombox[2],zoombox[3])
    plt.title('Zoom of BDF Stability Regions (shaded = unstable)')
    plt.savefig('BDF_stability_zoom.pdf')

    plt.show()


# end of script
