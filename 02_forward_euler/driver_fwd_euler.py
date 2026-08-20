#!/usr/bin/env python3
#
# Main routine to test the forward Euler method on two scalar-valued ODE problems
#    y' = -y, t in [0,5],
#    y(0) = 1.
# and
#    y' = (y+t^2-2)/(t+1), t in [0,5],
#    y(0) = 2.
#
# D.R. Reynolds
# Math 6321 @ SMU
# Fall 2023

import numpy as np
from ForwardEuler import *

# problem time interval
t0 = 0.0
tf = 5.0

# problem-definining functions and initial conditions
def f1(t,y):
    """ ODE RHS function """
    return -y
def ytrue1(t):
    """ Analytical solution """
    return np.array([np.exp(-t)])

def f2(t,y,alpha,beta):
    """ ODE RHS function (with parameters alpha and beta)"""
    return np.array([(alpha*y[0]+t*t-2.0*beta)/(t+1.0)])
def ytrue2(t):
    """ Analytical solution """
    return np.array([t*t + 2.0*t + 2.0 - 2.0*(t+1.0)*np.log(t+1.0)])

# shared testing data
Nout = 6   # includes initial condition
tspan = np.linspace(t0, tf, Nout)

# create true solution results
Y1true = np.zeros((Nout,1))
Y2true = np.zeros((Nout,1))
for i in range(Nout):
    Y1true[i,:] = ytrue1(tspan[i])
    Y2true[i,:] = ytrue2(tspan[i])

# time steps to try
hvals = np.array([0.5, 0.05, 0.005, 0.0005, 0.00005])
errs = np.zeros(hvals.size)

# problem 1: loop over time step sizes; call stepper and compute errors
print("\nProblem 1:")
FE1 = ForwardEuler(f1)
for idx, h in enumerate(hvals):

    # set initial condition and call stepper
    y0 = Y1true[0,:]
    print("  h = ", h, ":")
    FE1.reset()
    Y, success = FE1.Evolve(tspan, y0, h)

    # output solution, errors, and overall error
    Yerr = np.abs(Y-Y1true)
    errs[idx] = np.linalg.norm(Yerr,np.inf)
    for i in range(Nout):
        print("    y(%.1f) = %9.6f   |error| = %.2e" % (tspan[i], Y[i,0], Yerr[i,0]))
    print("  overall:  steps = %5i  abserr = %9.2e\n" % (FE1.get_num_steps(), errs[idx]))
orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
print('estimated order: max = ', np.max(orders), ',  avg = ', np.average(orders))


# problem 2: loop over time step sizes; call stepper and compute errors
print("\nProblem 2:")
FE2 = ForwardEuler(f2)
for idx, h in enumerate(hvals):

    # set initial condition and call stepper
    y0 = Y2true[0,:]
    print("  h = ", h, ":")
    FE2.reset()
    alpha = 1.0
    beta = 1.0
    # Here when calling our rhs (f2) we have two parameters alpha and beta.
    # Note that within ForwardEuler.py we call f(t,y,*args) -- here *(<tuple>) is the unpacking operator.
    # This will unpack the args tuple into the third (and any remaining) arguments of f2.
    Y, success = FE2.Evolve(tspan, y0, h, args=(alpha, beta))

    # output solution, errors, and overall error
    Yerr = np.abs(Y-Y2true)
    errs[idx] = np.linalg.norm(Yerr,np.inf)
    for i in range(Nout):
        print("    y(%.1f) = %9.6f   |error| = %.2e" % (tspan[i], Y[i,0], Yerr[i,0]))
    print("  overall:  steps = %5i  abserr = %9.2e  relerr = %9.2e\n" %
          (FE2.get_num_steps(), errs[idx], np.linalg.norm(Yerr/Y2true,np.inf)))
orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
print('estimated order: max = ', np.max(orders), ',  avg = ', np.average(orders))
