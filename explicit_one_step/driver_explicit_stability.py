#!/usr/bin/env python
#
# Script to test the forward Euler and some fixed-step ERK methods on the
# Dahlquist test problem
#     y' = lambda*y, t in [0,0.5],
#     y(0) = 1,
# for lambda = -100, h in {0.005, 0.01, 0.02, 0.04}
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
from Taylor2 import *
from ERK import *
from termcolor import colored

# problem time interval and Dahlquist parameter
t0 = 0.0
tf = 0.6
lam = -70.0

# problem-defining functions
def ytrue(t):
    """ Analytical solution """
    return np.array([np.exp(lam*t)])
def f(t,y):
    """ ODE RHS function """
    return (lam*y)
def f_t(t,y):
    """ t-derivative of ODE RHS function """
    return 0.0
def f_y(t,y):
    """ y-derivative of ODE RHS function """
    return (lam*np.array([[1.0]]))

# shared testing data
Nout = 6    # includes initial condition
tspan = np.linspace(t0, tf, Nout)
hvals = np.array([0.01, 0.02, 0.03, 0.04])

# create true solution results
Ytrue = np.zeros((Nout,1))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])

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
            text = "    y(%.3f) = %9.6f   |error| = %.2e" % (tspan[i], Y[i,0], Yerr[i,0])
            if Yerr[i,0] > 1.0:
                print(colored(text, "light_red"))
            else:
                print(text)
        print("  overall:  steps = %5i  abserr = %9.2e\n" % (stepper.get_num_steps(), np.linalg.norm(Yerr,np.inf)))

# Forward Euler
FE = ERK(f, ERK1())
print(colored("\nForward Euler:", "yellow", attrs=["bold"]))
run_stepper(FE, hvals, Ytrue, tspan)

# Taylor 2
T2 = Taylor2(f, f_t, f_y)
print(colored("\n2nd order Taylor:", "yellow", attrs=["bold"]))
run_stepper(T2, hvals, Ytrue, tspan)

# RK4
RK4 = ERK(f, ERK4())
print(colored("\n4th order explicit Runge-Kutta:", "yellow", attrs=["bold"]))
run_stepper(RK4, hvals, Ytrue, tspan)
