#!/usr/bin/env python
#
# Main routine to demonstrate the effect of the ARK coupling conditions,
# using pairs of explicit and implicit Butcher tables that are each
# accurate on their own, applied to the nonstiff additively-split problem
#    y' = fE(t,y) + fI(t,y),  t in [0,2],
#    y(0) = [1, 1/2],
# where
#    fE(t,y) = [cos(t) y_2^2, -sin(t) y_1]   (nonlinear, treated explicitly)
#    fI(t,y) = lambda*y,  lambda = -5          (linear, treated implicitly)
#
# The four pairs are:
#    Heun + implicit midpoint:  each 2nd order, but c^E != c^I, so the
#                               2nd-order coupling conditions fail (order 1)
#    ARS(1,2,2):                explicit + implicit midpoint, shared c (order 2)
#    RK4 + ESDIRK3:             shared c, each at least 3rd order, but the
#                               3rd-order coupling conditions fail (order 2)
#    ERK3 + ESDIRK3:            same ESDIRK and c, with an ERK whose b
#                               matches the ESDIRK (order 3)
#
# Since fI is linear, each implicit stage requires only a linear solve.
# This driver uses its own small fixed-step ARK routine; once ARK.py is
# added to this folder, that routine should be replaced by the ARK class.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
from scipy.integrate import solve_ivp

# problem time interval and parameters
t0 = 0.0
tf = 2.0
lam = -5.0
y0 = np.array([1.0, 0.5])

# problem-defining functions
def fE(t,y):
    """ Explicit (nonstiff) portion of the right-hand side """
    return np.array( [np.cos(t)*y[1]**2, -np.sin(t)*y[0]] )
def fI(t,y):
    """ Implicit portion of the right-hand side """
    return lam*y

# reference solution
ref = solve_ivp(lambda t,y: fE(t,y)+fI(t,y), (t0,tf), y0, method='DOP853',
                rtol=1e-13, atol=1e-14).y[:,-1]

# Butcher tables
def Heun():
    A = np.array(((0.0, 0.0), (1.0, 0.0)))
    b = np.array((0.5, 0.5))
    return {'A': A, 'b': b, 'c': np.sum(A,1)}
def ExplicitMidpoint():
    A = np.array(((0.0, 0.0), (0.5, 0.0)))
    b = np.array((0.0, 1.0))
    return {'A': A, 'b': b, 'c': np.sum(A,1)}
def ImplicitMidpointPadded():
    A = np.array(((0.0, 0.0), (0.0, 0.5)))
    b = np.array((0.0, 1.0))
    return {'A': A, 'b': b, 'c': np.sum(A,1)}
def RK4():
    A = np.array(((0.0, 0.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.0),
                  (0.0, 0.5, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)))
    b = np.array((1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0))
    return {'A': A, 'b': b, 'c': np.sum(A,1)}
def ERK3():
    A = np.array(((0.0, 0.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.0),
                  (0.0, 0.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    b = np.array((1.0/6.0, 0.0, 2.0/3.0, 1.0/6.0))
    return {'A': A, 'b': b, 'c': np.sum(A,1)}
def ESDIRK3():
    A = np.array(((0.0, 0.0, 0.0, 0.0), (1.0/6.0, 1.0/3.0, 0.0, 0.0),
                  (0.5, -1.0/3.0, 1.0/3.0, 0.0), (-2.0/3.0, 2.0/3.0, 2.0/3.0, 1.0/3.0)))
    b = np.array((1.0/6.0, 0.0, 2.0/3.0, 1.0/6.0))
    return {'A': A, 'b': b, 'c': np.sum(A,1)}

def ark_evolve(BE, BI, h):
    """
    Usage: y = ark_evolve(BE, BI, h)

    Fixed-step ARK evolution of the test problem over [t0,tf], using the
    explicit table BE for fE and the implicit table BI for the linear fI.
    """
    AE, bE, cE = BE['A'], BE['b'], BE['c']
    AI, bI, cI = BI['A'], BI['b'], BI['c']
    s = bE.size
    N = int(round((tf-t0)/h))
    t = t0
    y = y0.copy()
    for n in range(N):
        FE = []
        FI = []
        for i in range(s):
            a = y.copy()
            for j in range(i):
                a += h*(AE[i,j]*FE[j] + AI[i,j]*FI[j])
            z = a / (1.0 - h*AI[i,i]*lam)
            FE.append(fE(t+cE[i]*h, z))
            FI.append(fI(t+cI[i]*h, z))
        for j in range(s):
            y += h*(bE[j]*FE[j] + bI[j]*FI[j])
        t += h
    return y

# test runner function
hvals = 0.1 / 2.0**np.arange(6)
errs = np.zeros(hvals.size)
def RunTest(BE, BI, name):
    print("\n", name, " tests:", sep='')
    for idx, h in enumerate(hvals):
        errs[idx] = np.linalg.norm(ark_evolve(BE, BI, h) - ref, np.inf)
        print("    h = %.5f:  abserr = %8.2e" % (h, errs[idx]))
    orders = np.log(errs[0:-1]/errs[1:])/np.log(hvals[0:-1]/hvals[1:])
    print('    estimated order:  max = %.2f,  avg = %.2f' %
          (np.max(orders), np.average(orders)))

RunTest(Heun(), ImplicitMidpointPadded(), 'Heun + implicit midpoint')
RunTest(ExplicitMidpoint(), ImplicitMidpointPadded(), 'ARS(1,2,2)')
RunTest(RK4(), ESDIRK3(), 'RK4 + ESDIRK3')
RunTest(ERK3(), ESDIRK3(), 'ERK3 + ESDIRK3')
