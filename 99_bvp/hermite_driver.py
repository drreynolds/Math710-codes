#!/usr/bin/env python
#
# Main routine to run a piecewise Hermite finite-difference method
# for solution of a second-order, scalar-valued BVP:
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

# utility routine to map from physical/component space to linear algebra index space
#    interval:  physical interval index [1 <= interval <= N]
#    location:  location in interval [0=left, 1=right]
#    component: solution component at this location [0=u, 1=u']
def index(interval, location, component):
    return ( 2*(interval-1) + 2*location + component )

# utility routines for Hermite basis functions (and corresponding derivatives)
def phi1(tleft, h, t):
    return (2*((t-tleft)/h)**3 - 3*((t-tleft)/h)**2 + 1)
def dphi1(tleft, h, t):
    return (6/h*((t-tleft)/h)**2 - 6*(t-tleft)/h/h)
def ddphi1(tleft, h, t):
    return (12*(t-tleft)/h/h/h - 6/h/h)
def phi2(tleft, h, t):
    return (h*((t-tleft)/h)**3 - 2*h*((t-tleft)/h)**2 + (t-tleft))
def dphi2(tleft, h, t):
    return (3*((t-tleft)/h)**2 - 4*(t-tleft)/h + 1)
def ddphi2(tleft, h, t):
    return (6*(t-tleft)/h/h - 4/h)
def phi3(tleft, h, t):
    return (-2*((t-tleft)/h)**3 + 3*((t-tleft)/h)**2)
def dphi3(tleft, h, t):
    return (-6/h*((t-tleft)/h)**2 + 6*(t-tleft)/h/h)
def ddphi3(tleft, h, t):
    return (-12*(t-tleft)/h/h/h + 6/h/h)
def phi4(tleft, h, t):
    return (h*((t-tleft)/h)**3 - h*((t-tleft)/h)**2)
def dphi4(tleft, h, t):
    return (3*((t-tleft)/h)**2 - 2*(t-tleft)/h)
def ddphi4(tleft, h, t):
  return (6*(t-tleft)/h/h - 2/h)

# get lambda from the command line, otherwise set to -10
lam = -10.0
if (len(sys.argv) > 1):
    lam = float(sys.argv[1])

# create BVP object
bvp = BVP(lam)

# test 'index' function by outputting mapping for small N
print ("Test output from 'index' function for N = 3, M = 8:")
for interval in range(1,4):
    print("\n  interval ", interval, ", (loc,comp,idx):")
    for location in range(2):
        for component in range(2):
            print("  (", location, ", ", component, ", ", index(interval,location,component), ")")


# test basis functions
#   check {1,0} properties
delta = 1e-8
tl = 0.5
tr = 0.6
tt = 0.53
h = tr-tl
print("\n\nOsculatory interpolation tests:")
failed = False
dtest = phi1(tl,h,tl) - 1
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi1(tleft) error, value = ", dtest + 1)
dtest = phi1(tl,h,tr)
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi1(tright) error, value = ", dtest)
dtest = (phi1(tl,h,tl+delta) - phi1(tl,h,tl))/delta
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi1'(tleft) error, value = ", dtest)
dtest = (phi1(tl,h,tr+delta) - phi1(tl,h,tr))/delta
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi1'(tright) error, value = ", dtest)
dtest = phi2(tl,h,tl)
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi2(tleft) error, value = ", dtest)
dtest = phi2(tl,h,tr)
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi2(tright) error, value = ", dtest)
dtest = (phi2(tl,h,tl+delta) - phi2(tl,h,tl))/delta - 1
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi2'(tleft) error, value = ", dtest + 1)
dtest = (phi2(tl,h,tr+delta) - phi2(tl,h,tr))/delta
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi2'(tright) error, value = ", dtest)
dtest = phi3(tl,h,tl)
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi3(tleft) error, value = ", dtest)
dtest = phi3(tl,h,tr) - 1
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi3(tright) error, value = ", dtest + 1)
dtest = (phi3(tl,h,tl+delta) - phi3(tl,h,tl))/delta
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi3'(tleft) error, value = ", dtest)
dtest = (phi3(tl,h,tr+delta) - phi3(tl,h,tr))/delta
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi3'(tright) error, value = ", dtest)
dtest = phi4(tl,h,tl)
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi4(tleft) error, value = ", dtest)
dtest = phi4(tl,h,tr)
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi4(tright) error, value = ", dtest)
dtest = (phi4(tl,h,tl+delta) - phi4(tl,h,tl))/delta
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi4'(tleft) error, value = ", dtest)
dtest = (phi4(tl,h,tr+delta) - phi4(tl,h,tr))/delta - 1
if (np.abs(dtest) > 1e-4):
    failed = True
    print("    phi4'(tright) error, value = ", dtest + 1)
if (not failed):
    print( "  all tests pass")

#   check analytical derivative tests
print( "Derivative tests:")
failed = False
dtest = (phi1(tl,h,tt+delta)-phi1(tl,h,tt))/delta
if (np.abs(dtest - dphi1(tl,h,tt)) > 1e-4):
    failed = True
    print("  dphi1 error, value = ", dphi1(tl,h,tt), ", approx = ", dtest)
dtest = (dphi1(tl,h,tt+delta)-dphi1(tl,h,tt))/delta
if (np.abs(dtest - ddphi1(tl,h,tt)) > 1e-4):
    failed = True
    print( "  ddphi1 error, value = ", ddphi1(tl,h,tt), ", approx = ", dtest )
dtest = (phi2(tl,h,tt+delta)-phi2(tl,h,tt))/delta
if (np.abs(dtest - dphi2(tl,h,tt)) > 1e-4):
    failed = True
    print( "  dphi2 error, value = ", dphi2(tl,h,tt), ", approx = ", dtest )
dtest = (dphi2(tl,h,tt+delta)-dphi2(tl,h,tt))/delta
if (np.abs(dtest - ddphi2(tl,h,tt)) > 1e-4):
    failed = True
    print( "  ddphi2 error, value = ", ddphi2(tl,h,tt), ", approx = ", dtest )
dtest = (phi3(tl,h,tt+delta)-phi3(tl,h,tt))/delta
if (np.abs(dtest - dphi3(tl,h,tt)) > 1e-4):
    failed = True
    print( "  dphi3 error, value = ", dphi3(tl,h,tt), ", approx = ", dtest )
dtest = (dphi3(tl,h,tt+delta)-dphi3(tl,h,tt))/delta
if (np.abs(dtest - ddphi3(tl,h,tt)) > 1e-4):
    failed = True
    print( "  ddphi3 error, value = ", ddphi3(tl,h,tt), ", approx = ", dtest )
dtest = (phi4(tl,h,tt+delta)-phi4(tl,h,tt))/delta
if (np.abs(dtest - dphi4(tl,h,tt)) > 1e-4):
    failed = True
    print( "  dphi4 error, value = ", dphi4(tl,h,tt), ", approx = ", dtest )
dtest = (dphi4(tl,h,tt+delta)-dphi4(tl,h,tt))/delta
if (np.abs(dtest - ddphi4(tl,h,tt)) > 1e-4):
    failed = True
    print( "  ddphi4 error, value = ", ddphi4(tl,h,tt), ", approx = ", dtest )
if (not failed):
    print( "  all tests pass")


# loop over spatial resolutions for tests
N = [100, 1000, 10000]
for n in N:

    # output problem information
    print("\nPiecewise Hermite FD method for BVP with lambda = %.1f,  N = %i" % (lam, n))

    # compute/store analytical solution
    t = np.zeros(n+1)
    t[0] = 0
    t[n] = 1
    for j in range(1,n):
        t[j] = 0.5*(1-np.cos((2*j-1)*np.pi/(2*(n-1))))
    utrue = np.zeros(n+1)
    for j in range(n+1):
        utrue[j] = bvp.utrue(t[j])

    # set integer for overall linear algebra problem size
    M = 2*n+2

    # create matrix and right-hand side vectors
    Arows = np.zeros(8*n+2)
    Acols = np.zeros(8*n+2)
    Avals = np.zeros(8*n+2)
    b = np.zeros(M)

    # set up linear system:
    #    recall 'index' usage: index(interval,location,component)
    #      interval:  physical interval index [1 <= interval <= N]
    #      location:  location in interval [0=left, 1=right]
    #      component: solution component at this location [0=u, 1=u']
    #    recall [dd]phiN usage: [dd]phiN(tleft, h, t)
    idx = 0
    Arows[idx] = 0        # A(0,index(1,0,0))
    Acols[idx] = index(1,0,0)
    Avals[idx] = 1
    idx += 1
    b[0] = bvp.ua

    Arows[idx] = 1        # A(1,index(n,1,0))
    Acols[idx] = index(n,1,0)
    Avals[idx] = 1
    idx += 1
    b[1] = bvp.ub

    irow = 2
    for j in range(1,n+1):

      # setup interval-specific information
      tl = t[j-1]
      tr = t[j]
      h = tr-tl
      eta1 = 0.5*(tr+tl) - h/2/np.sqrt(3)
      eta2 = 0.5*(tr+tl) + h/2/np.sqrt(3)
      q1 = bvp.q(eta1)
      q2 = bvp.q(eta2)
      p1 = bvp.p(eta1)
      p2 = bvp.p(eta2)

      # setup first equation for this interval: enforce ODE at eta1
      Arows[idx] = irow        # A(irow,index(j,0,0))
      Acols[idx] = index(j,0,0)
      Avals[idx] = ddphi1(tl,h,eta1) - p1*dphi1(tl,h,eta1) - q1*phi1(tl,h,eta1)
      idx += 1

      Arows[idx] = irow        # A(irow,index(j,0,1))
      Acols[idx] = index(j,0,1)
      Avals[idx] = ddphi2(tl,h,eta1) - p1*dphi2(tl,h,eta1) - q1*phi2(tl,h,eta1)
      idx += 1

      Arows[idx] = irow        # A(irow,index(j,1,0))
      Acols[idx] = index(j,1,0)
      Avals[idx] = ddphi3(tl,h,eta1) - p1*dphi3(tl,h,eta1) - q1*phi3(tl,h,eta1)
      idx += 1

      Arows[idx] = irow        # A(irow,index(j,1,1))
      Acols[idx] = index(j,1,1)
      Avals[idx] = ddphi4(tl,h,eta1) - p1*dphi4(tl,h,eta1) - q1*phi4(tl,h,eta1)
      idx += 1

      b[irow] = bvp.r(eta1)
      irow += 1

      # setup second equation for this interval: enforce ODE at eta1
      Arows[idx] = irow        # A(irow,index(j,0,0))
      Acols[idx] = index(j,0,0)
      Avals[idx] = ddphi1(tl,h,eta2) - p2*dphi1(tl,h,eta2) - q2*phi1(tl,h,eta2)
      idx += 1

      Arows[idx] = irow        # A(irow,index(j,0,1))
      Acols[idx] = index(j,0,1)
      Avals[idx] = ddphi2(tl,h,eta2) - p2*dphi2(tl,h,eta2) - q2*phi2(tl,h,eta2)
      idx += 1

      Arows[idx] = irow        # A(irow,index(j,1,0))
      Acols[idx] = index(j,1,0)
      Avals[idx] = ddphi3(tl,h,eta2) - p2*dphi3(tl,h,eta2) - q2*phi3(tl,h,eta2)
      idx += 1

      Arows[idx] = irow        # A(irow,index(j,1,1))
      Acols[idx] = index(j,1,1)
      Avals[idx] = ddphi4(tl,h,eta2) - p2*dphi4(tl,h,eta2) - q2*phi4(tl,h,eta2)
      idx += 1

      b[irow] = bvp.r(eta2)
      irow += 1

    A = sp.csr_matrix(sp.coo_matrix((Avals, (Arows, Acols)), shape=(M, M)))

    # solve linear system for BVP solution
    y = spsolve(A,b)

    # output maximum error
    u = np.zeros(n+1)
    for j in range(n+1):
        u[j] = y[index(j+1,0,0)]
    uerr = np.abs(u-utrue)
    print("  Maximum BVP solution error = %.4e" % (np.max(uerr)))
