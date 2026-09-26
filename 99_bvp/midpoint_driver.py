#!/usr/bin/env python
#
# Main routine to run an implicit midpoint finite-difference method for solution
# of a second-order, scalar-valued BVP:
#
#    u'' = p(t)*u' + q(t)*u + r(t),  a<t<b,
#    u(a) = ua,  u(b) = ub
#
# where the problem has stiffness that may be adjusted using
# the real-valued parameter lambda<0 [read from the command line]
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
    print("\nImplicit Midpoint FD method for BVP with lambda = %.1f,  N = %i" % (lam, n))

    # compute/store analytical solution
    t = np.zeros(n+1)
    t[0] = 0.0
    t[n] = 1.0
    for j in range(1,n):
        t[j] = 0.5*(1-np.cos((2*j-1)*np.pi/(2*(n-1))))
    utrue = np.zeros(n+1)
    for j in range(n+1):
      utrue[j] = bvp.utrue(t[j])

    # create matrix and right-hand side vectors
    Arows = np.zeros(8*(n+1))
    Acols = np.zeros(8*(n+1))
    Avals = np.zeros(8*(n+1))
    b = np.zeros(2*(n+1))

    # set up linear system:
    #   note: y = [y_{0,1} y_{0,2} y_{1,1} y_{1,2} ... y_{N,1} y_{N,2}]
    idx = 0
    Arows[idx] = 0   # A(0,0) = 1.0
    Acols[idx] = 0
    Avals[idx] = 1
    idx += 1
    b[0] = bvp.ua

    Arows[idx] = 1   # A(1,2*n) = 1.0
    Acols[idx] = 2*n
    Avals[idx] = 1
    idx += 1
    b[1] = bvp.ub

    for j in range(1,n+1):

        # setup interval-specific information
        h = t[j]-t[j-1]
        thalf = 0.5*(t[j]+t[j-1])
        alpha = -h*bvp.q(thalf)
        beta = h*bvp.p(thalf)
        gamma = 2*h*bvp.r(thalf)

        # setup eqn in row 2*j:
        #    2*y_{j,1} - 2*y_{j-1,1} - h*y_{j,2} - h*y_{j-1,2} = 0
        Arows[idx] = 2*j      # A(2*j,2*j)
        Acols[idx] = 2*j
        Avals[idx] = 2.0
        idx += 1

        Arows[idx] = 2*j      # A(2*j,2*j-2)
        Acols[idx] = 2*j-2
        Avals[idx] = -2.0
        idx += 1

        Arows[idx] = 2*j      # A(2*j,2*j+1)
        Acols[idx] = 2*j+1
        Avals[idx] = -h
        idx += 1

        Arows[idx] = 2*j      # A(2*j,2*j-1)
        Acols[idx] = 2*j-1
        Avals[idx] = -h
        idx += 1

        b[2*j] = 0.0

        # setup eqn in row 2*j+1:
        #     alpha*y_{j,1} + alpha*y_{j-1,1} + (2-beta)*y_{j,2} - (2+beta)*y_{j-1,2} = gamma_j
        Arows[idx] = 2*j+1    # A(2*j+1,2*j)
        Acols[idx] = 2*j
        Avals[idx] = alpha
        idx += 1

        Arows[idx] = 2*j+1    # A(2*j+1,2*j-2)
        Acols[idx] = 2*j-2
        Avals[idx] = alpha
        idx += 1

        Arows[idx] = 2*j+1     # A(2*j+1,2*j+1)
        Acols[idx] = 2*j+1
        Avals[idx] = 2.0-beta
        idx += 1

        Arows[idx] = 2*j+1    # A(2*j+1,2*j-1)
        Acols[idx] = 2*j-1
        Avals[idx] = -(2.0+beta)
        idx += 1

        b[2*j+1] = gamma

    A = sp.csr_matrix(sp.coo_matrix((Avals, (Arows, Acols)), shape=(2*(n+1), 2*(n+1))))

    # solve linear system for BVP solution
    y = spsolve(A,b)

    # output maximum error
    u = np.zeros(n+1)
    for j in range(n+1):
        u[j] = y[2*j]
    uerr = np.abs(u-utrue)
    print("  Maximum BVP solution error = %.4e" % (np.max(uerr)))
