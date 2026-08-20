#!/usr/bin/env python3
#
# Main routine to test adaptive forward Euler method on a system of ODEs
#    y' = f(t,y), t in [0,1],
#    y(0) = y0.
#
# D.R. Reynolds
# Math 6321 @ SMU
# Fall 2023

import numpy as np
import sys
from AdaptEuler import *

# get problem size from command line, otherwise set to 5
N = 5
if (len(sys.argv) > 1):
    N = int(sys.argv[1])
print("\nRunning system ODE problem with N = ", N)

# set up problem data
V = np.eye(N) + np.random.random_sample((N,N))  # fill V,D with random numbers
D = np.diag(-np.random.random_sample(N))
Vinv = np.linalg.inv(V)                         # Vinv = V^{-1}
A = V @ D @ Vinv                                # construct system matrix
if (N < 10):
    print("\nProblem-defining matrices:")
    print("V:\n", V)
    print("Vinv:\n", Vinv)
    print("D:\n", D)
    print("A:\n", A)

# set problem time interval and initial condition
t0 = 0.0
tf = 1.0
y0 = np.random.random_sample(N)

# problem-defining functions
def f(t,y):
    """ ODE RHS function """
    return A@y
def ytrue(t):
    """ Analytical solution """
    eD = np.zeros((N,N))       # construct the matrix exponential
    for i in range(N):
        eD[i,i] = np.exp(D[i,i]*(t-t0))
    return (V @ (eD @ (Vinv @ y0)))  # ytrue = V exp(D*t) V^{-1} y0

# shared testing data
Nout = 6   # includes initial condition
tspan = np.linspace(t0, tf, Nout)

# get true solution at output times
Ytrue = np.zeros((Nout,N))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])

# set the testing tolerances
rtols = np.array((1.e-3, 1.e-5, 1.e-7))
atol = 1.e-13

# create adaptive forward Euler stepper object (will reset rtol before each solve)
AE = AdaptEuler(f, y0, atol=atol)

# loop over relative tolerances
print("\nAdaptive Euler test problem, steps and errors vs tolerances:")
for rtol in rtols:

    # set the relative tolerance, and call the solver
    print("  rtol = ", rtol)
    AE.set_rtol(rtol)
    AE.reset()
    Y, success = AE.Evolve(tspan, y0)
    if (not success):
        print("    solve failed at this tolerance")
        continue

    # output solution, errors, and overall error
    Yerr = np.abs(Y-Ytrue)
    if (N < 10):
        for i in range(Nout):
            print("    y(%.1f) = " % (tspan[i]), Y[i,:])
    print("  overall:  steps = %5i  fails = %2i  abserr = %9.2e  relerr = %9.2e\n" %
          (AE.get_num_steps(), AE.get_num_error_failures(), np.linalg.norm(Yerr,np.inf),
           np.linalg.norm(Yerr/Ytrue,np.inf)))
