#!/usr/bin/env python
#
# Main routine to test the higher-order one-step methods.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
from Taylor2 import *
from ERK import *

# problem time interval
t0 = 0.0
tf = 1.0

# problem-definining functions and initial conditions
def f(t,y):
    """ ODE RHS function """
    return -y*np.exp(-t)
def f_t(t,y):
    """ t-derivative of ODE RHS function """
    return y*np.exp(-t)
def f_y(t,y):
    """ y-derivative of ODE RHS function """
    return np.array([[-np.exp(-t)]])
def ytrue(t):
    """ Analytical solution """
    return np.exp(np.exp(-t)-1.0)

# shared testing data
Nout = 3   # includes initial condition
tspan = np.linspace(t0, tf, Nout)

# create true solution results
Ytrue = np.zeros((Nout,1))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])

# time steps to try
hvals = np.array([0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005])
errs = np.zeros(hvals.size)

# utility routine to run a given stepper and compute errors
def run_stepper(stepper, hvals, Ytrue, tspan):
    """
    Runs a given stepper on the test problem for a range of time step sizes,
    and computes the errors.
    """
    errs = np.zeros(hvals.size)
    for idx, h in enumerate(hvals):

        # set initial condition and call stepper
        y0 = Ytrue[0,:]
        print("  h = ", h, ":")
        stepper.reset()
        Y, success = stepper.Evolve(tspan, y0, h)

        # output solution, errors, and overall error
        Yerr = np.abs(Y-Ytrue)
        errs[idx] = np.linalg.norm(Yerr,np.inf)
        for i in range(Nout):
            print("    y(%.1f) = %9.6f   |error| = %.2e" % (tspan[i], Y[i,0], Yerr[i,0]))
        print("  overall:  steps = %5i  nrhs = %5i  abserr = %9.2e  relerr = %9.2e\n" %
              (stepper.get_num_steps(), stepper.get_num_rhs(), errs[idx], np.linalg.norm(Yerr/Ytrue,np.inf)))
    orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
    print('estimated order: max = ', np.max(orders), ',  avg = ', np.average(orders))

#### ERK1 ####
print("\nERK1:")
FE = ERK(f, ERK1())
run_stepper(FE, hvals, Ytrue, tspan)

#### Taylor 2 ####
print("\nTaylor2:")
T2 = Taylor2(f, f_t, f_y)
run_stepper(T2, hvals, Ytrue, tspan)

#### Heun ####
print("\nHeun:")
H = ERK(f, Heun())
run_stepper(H, hvals, Ytrue, tspan)

#### ERK2 ####
print("\nERK2:")
E2 = ERK(f, ERK2())
run_stepper(E2, hvals, Ytrue, tspan)

#### ERK3 ####
print("\nERK3:")
E3 = ERK(f, ERK3())
run_stepper(E3, hvals, Ytrue, tspan)

#### ERK4 ####
print("\nERK4:")
E4 = ERK(f, ERK4())
run_stepper(E4, hvals, Ytrue, tspan)
