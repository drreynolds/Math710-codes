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
    print("\nStencil-based FD method for BVP with lambda =", lam, ",  N =", n)

    # compute/store analytical solution
    t = np.linspace(bvp.a, bvp.b, n+1)
    h = t[1]-t[0]
    utrue = np.zeros(n+1)
    for j in range(n+1):
      utrue[j] = bvp.utrue(t[j])

    # create matrix and right-hand side vectors
    Arows = np.zeros(3*(n+1)-4)
    Acols = np.zeros(3*(n+1)-4)
    Avals = np.zeros(3*(n+1)-4)
    b = np.zeros(n+1)

    # set up linear system
    idx = 0
    Arows[idx] = 0   # A(0,0) = 1.0
    Acols[idx] = 0
    Avals[idx] = 1
    idx += 1
    b[0] = bvp.ua

    Arows[idx] = n   # A(n,n) = 1.0
    Acols[idx] = n
    Avals[idx] = 1
    idx += 1
    b[n] = bvp.ub

    for j in range(1,n):
        Arows[idx] = j     # A[j,j-1]
        Acols[idx] = j-1
        Avals[idx] = -1 - 0.5*h*bvp.p(t[j])
        idx += 1
        Arows[idx] = j     # A[j,j]
        Acols[idx] = j
        Avals[idx] = 2 + h*h*bvp.q(t[j])
        idx += 1
        Arows[idx] = j     # A[j,j+1]
        Acols[idx] = j+1
        Avals[idx] = -1 + 0.5*h*bvp.p(t[j])
        idx += 1
        b[j] = -h*h*bvp.r(t[j])
    A = sp.csr_matrix(sp.coo_matrix((Avals, (Arows, Acols)), shape=(n+1, n+1)))

    # solve linear system for BVP solution
    u = spsolve(A,b)

    # output maximum error
    uerr = np.abs(u-utrue)
    print("  Maximum BVP solution error = %.4e" % (np.max(uerr)))
