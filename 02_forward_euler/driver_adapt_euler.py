#!/usr/bin/env python3
#
# Main routine to test adaptive forward Euler method on the scalar-valued ODE problem
#    y' = -exp(-t)*y, t in [0,5],
#    y(0) = 1.
#
# D.R. Reynolds
# Math 6321 @ SMU
# Fall 2023

import numpy as np
from AdaptEuler import *

# problem time interval
t0 = 0.0
tf = 5.0

# problem-definining functions
def f(t,y,alpha):
    """ ODE RHS function (with example parameter alpha) """
    return np.array([-alpha*np.exp(-t)*y[0]])
def ytrue(t):
    """ Analytical solution """
    return np.array([np.exp(np.exp(-t)-1.0)])

# shared testing data
Nout = 6   # includes initial condition
tspan = np.linspace(t0, tf, Nout)

# get true solution at output times
Ytrue = np.zeros((Nout,1))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])
y0 = Ytrue[0,:]

# set the testing tolerances
rtols = np.array((1.e-3, 1.e-5, 1.e-7))
atol = 1.e-11

# create adaptive forward Euler stepper object (will reset rtol before each solve)
AE = AdaptEuler(f, y0)

# loop over relative tolerances
print("Adaptive Euler test problem, steps and errors vs tolerances:")
for rtol in rtols:

    # set the relative tolerance, and call the solver
    print("  rtol = ", rtol)
    AE.set_rtol(rtol)
    alpha = 1.0
    # Here when calling our rhs we have a single parameter alpha.
    # Note that within AdaptEuler.py we call f(t,y,*args) -- here *(<tuple>) is the unpacking operator.
    # This will unpack the args tuple into the third (and any remaining) arguments of f.
    # Note that the "," here is required to ensure that args is an iterable (and not a float).
    Y, success = AE.Evolve(tspan, y0, args=(alpha,))
    if (not success):
        print("    solve failed at this tolerance")
        continue

    # output solution, errors, and overall error
    Yerr = np.abs(Y-Ytrue)
    for i in range(Nout):
        print("    y(%.1f) = %9.6f,   abserr = %.2e,  relerr = %.2e" %
              (tspan[i], Y[i,0], Yerr[i,0], Yerr[i,0]/np.abs(Y[i,0])))
    print("  overall:  steps = %5i  fails = %2i  abserr = %9.2e  relerr = %9.2e\n" %
          (AE.get_num_steps(), AE.get_num_error_failures(), np.linalg.norm(Yerr,np.inf),
           np.linalg.norm(Yerr/Ytrue,np.inf)))
