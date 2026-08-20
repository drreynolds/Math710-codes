#!/usr/bin/env python3
#
# Main routine to test the forward Euler method on a system of ODEs
#    y' = f(t,y), t in [0,1],
#    y(0) = y0.
#
# D.R. Reynolds
# Math 6321 @ SMU
# Fall 2023

import numpy as np
import sys
from ForwardEuler import *

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


# time steps to try
hvals = np.array([0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125])
errs = np.zeros(hvals.size)

# create forward Euler stepper object (will reset rtol before each solve)
FE = ForwardEuler(f)

# loop over time step sizes; call stepper and compute errors
for idx, h in enumerate(hvals):

    # set initial condition and call stepper
    print("\nRunning with stepsize h = ", h, ":")
    FE.reset()
    Y, success = FE.Evolve(tspan, y0, h)

    # output solution, errors, and overall error
    Yerr = np.abs(Y-Ytrue)
    errs[idx] = np.linalg.norm(Yerr,np.inf)
    if (N < 10):
        for i in range(Nout):
            print("    y(%.1f) = " % (tspan[i]), Y[i,:])
    print("  overall:  steps = %5i  abserr = %9.2e  relerr = %9.2e" %
          (FE.get_num_steps(), errs[idx], np.linalg.norm(Yerr/Ytrue,np.inf)))
orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
print('estimated order: max = ', np.max(orders), ',  avg = ', np.average(orders))
