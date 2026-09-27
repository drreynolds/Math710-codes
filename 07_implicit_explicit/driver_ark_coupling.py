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
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.append('..')
from shared.ImplicitSolver import *
from ARK import *

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
def JI(t,y):
    """ Jacobian of the implicit portion of the right-hand side """
    return lam*np.eye(y.size)

# reference solution
ref = solve_ivp(lambda t,y: fE(t,y)+fI(t,y), (t0,tf), y0, method='DOP853',
                rtol=1e-13, atol=1e-14).y[:,-1]

# test runner function
hvals = 0.1 / 2.0**np.arange(6)
errs = np.zeros(hvals.size)
def RunTest(BE, BI, name):
    print("\n", name, " tests:", sep='')
    solver = ImplicitSolver(JI, solver_type='dense', maxiter=8,
                            rtol=1e-12, atol=1e-14)
    stepper = ARK(fE, fI, solver, BE, BI)
    for idx, h in enumerate(hvals):
        stepper.reset()
        stepper.sol.reset()
        Y, success = stepper.Evolve(np.array([t0,tf]), y0, h)
        errs[idx] = np.linalg.norm(Y[-1,:] - ref, np.inf)
        if (success):
            print("    h = %.5f:  solves = %4i  abserr = %8.2e" %
                  (h, stepper.get_num_solves(), errs[idx]))
    orders = np.log(errs[0:-1]/errs[1:])/np.log(hvals[0:-1]/hvals[1:])
    print('    estimated order:  max = %.2f,  avg = %.2f' %
          (np.max(orders), np.average(orders)))

BE, BI = HeunImplicitMidpoint()
RunTest(BE, BI, 'Heun + implicit midpoint')
BE, BI = ARS122()
RunTest(BE, BI, 'ARS(1,2,2)')
BE, BI = RK4ESDIRK3()
RunTest(BE, BI, 'RK4 + ESDIRK3')
BE, BI = ERK3ESDIRK3()
RunTest(BE, BI, 'ERK3 + ESDIRK3')
