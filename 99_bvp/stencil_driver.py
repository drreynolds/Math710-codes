#!/usr/bin/env python
#
# Main routine to run a stencil-based finite-difference method for solution of a
# second-order, scalar-valued BVP:
#
#    u'' = p(t)*u' + q(t)*u + r(t),  a<t<b,
#    u(a) = ua,  u(b) = ub
#
# where the problem has stiffness that may be adjusted using
# the real-valued parameter lambda<0 [read from the command line]
#
# This driver attempts to solve the problem using a second-order, stencil-based
# finite-difference approximation.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from BVP import *

# get lambda from the command line, otherwise set to -10
lam = -10.0
if (len(sys.argv) > 1):
    lam = float(sys.argv[1])

# create BVP object
bvp = BVP(lam)

# loop over spatial resolutions for tests
N = [100, 1000, 10000]
for n in N:

    # output problem information
    print("\nStencil-based FD method for BVP with lambda = %.1f,  N = %i" % (lam, n))

    # compute/store analytical solution
    t = np.linspace(bvp.a, bvp.b, n+1)
    h = t[1]-t[0]
    utrue = bvp.utrue(t)

    # create matrix diagonals and right-hand side vector
    #   note: spdiags indexes each diagonal by column, so A[j,j-1] is stored
    #   in lower[j-1], A[j,j] in diagv[j], and A[j,j+1] in upper[j+1]
    lower = np.zeros(n+1)
    diagv = np.zeros(n+1)
    upper = np.zeros(n+1)
    b = np.zeros(n+1)

    # set up linear system
    diagv[0] = 1     # A(0,0) = 1.0
    b[0] = bvp.ua

    diagv[n] = 1     # A(n,n) = 1.0
    b[n] = bvp.ub

    j = np.arange(1,n)
    lower[j-1] = -1 - 0.5*h*bvp.p(t[j])     # A[j,j-1]
    diagv[j] = 2 + h*h*bvp.q(t[j])          # A[j,j]
    upper[j+1] = -1 + 0.5*h*bvp.p(t[j])     # A[j,j+1]
    b[j] = -h*h*bvp.r(t[j])
    A = sp.csr_matrix(sp.spdiags(np.vstack((lower, diagv, upper)), [-1, 0, 1], n+1, n+1))

    # solve linear system for BVP solution
    u = spsolve(A,b)

    # output maximum error
    uerr = np.abs(u-utrue)
    print("  Maximum BVP solution error = %.4e" % (np.max(uerr)))
