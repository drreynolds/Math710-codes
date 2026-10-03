#!/usr/bin/env python
#
# Main routine to test the backward Euler, trapezoidal, and
# forward Euler methods on the scalar-valued ODE problem
#    y' = lambda*y + (1-lambda)*cos(t) - (1+lambda)*sin(t), t in [0,5],
#    y(0) = 1.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC
import numpy as np
import sys
from termcolor import colored
sys.path.append('../shared')
from ImplicitSolver import *
sys.path.append('../02_forward_euler')
from ForwardEuler import *
from BackwardEuler import *
from Trapezoidal import *

# problem time interval and parameters
t0 = 0.0
tf = 5.0

# problem-defining functions
def ytrue(t):
    """ Generates a numpy array containing the true solution to the IVP at a given input t. """
    return np.array([np.sin(t) + np.cos(t)])
def f(t,y,lam):
    """ Right-hand side function, f(t,y), for the IVP """
    return np.array([lam*y[0] + (1.0-lam)*np.cos(t) - (1.0+lam)*np.sin(t)])
def J(t,y,lam):
    """ Jacobian (in dense matrix format) of the right-hand side function, J(t,y) = df/dy """
    return np.array( [ [lam] ] )

# construct implicit solvers
solver = ImplicitSolver(J, solver_type='dense', maxiter=20, rtol=1e-9, atol=1e-12, Jfreq=2)

# shared testing data
Nout = 6   # includes initial condition
tspan = np.linspace(t0, tf, Nout)
Ytrue = np.zeros((Nout, 1))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])
y0 = Ytrue[0,:]
hvals = np.array([1.0, 0.1, 0.01, 0.001])
errs = np.zeros(hvals.size)

# create forward Euler, backward Euler, and trapezoidal solvers
BE = BackwardEuler(f, solver)
Tr = Trapezoidal(f, solver)
FE = ForwardEuler(f)


# loop over stiffness values
for lam in [-1.0, -10.0, -50.0]:

    # backward Euler tests
    print(colored("\nbackward Euler tests:", "yellow", attrs=["bold"]))
    for idx, h in enumerate(hvals):
        print("  h = ",h,",  lambda = ", lam,":", sep='')
        BE.reset()
        BE.sol.reset()
        # Note that this is where we provide the rhs function parameter lam -- the "," is
        # required to ensure that args is an iterable (and not a float).
        Y, success = BE.Evolve(tspan, y0, h, args=(lam,))
        Yerr = np.abs(Y-Ytrue)
        errs[idx] = np.linalg.norm(Yerr,np.inf)
        if (success):
            print("     " + colored("  t      y(t)     |err(t)| ", attrs=["underline"]))
            for i in range(Nout):
                text = "      %.1f  %10.2e  %.2e" % (tspan[i], Y[i,0], Yerr[i,0])
                if (Yerr[i,0] > 1):
                    print(colored(text, "light_red"))
                else:
                    print(text)
            text = "  overall:  steps = %4i  Niters = %6i  NJevals = %5i  abserr = %8.2e" \
                % (BE.get_num_steps(), BE.sol.get_total_iters(),
                   BE.sol.get_total_setups(), errs[idx])
            if (errs[idx] > 1):
                print(colored(text, "red"))
            else:
                print(colored(text, "green"))
    orders = np.log(errs[:-1]/errs[1:])/np.log(hvals[:-1]/hvals[1:])
    print('estimated order: max = ', np.max(orders), ',  avg = ', np.average(orders))

    # trapezoidal tests
    print(colored("\ntrapezoidal tests:", "yellow", attrs=["bold"]))
    for idx, h in enumerate(hvals):
        print("  h = ",h,",  lambda = ", lam,":", sep='')
        Tr.reset()
        Tr.sol.reset()
        Y, success = Tr.Evolve(tspan, y0, h, args=(lam,))
        Yerr = np.abs(Y-Ytrue)
        errs[idx] = np.linalg.norm(Yerr,np.inf)
        if (success):
            print("     " + colored("  t      y(t)     |err(t)| ", attrs=["underline"]))
            for i in range(Nout):
                text = "      %.1f  %10.2e  %.2e" % (tspan[i], Y[i,0], Yerr[i,0])
                if (Yerr[i,0] > 1):
                    print(colored(text, "light_red"))
                else:
                    print(text)
            text = "  overall:  steps = %4i  Niters = %6i  NJevals = %5i  abserr = %8.2e" % \
                  (Tr.get_num_steps(), Tr.sol.get_total_iters(),
                   Tr.sol.get_total_setups(), errs[idx])
            if (errs[idx] > 1):
                print(colored(text, "red"))
            else:
                print(colored(text, "green"))
    orders = np.log(errs[:-1]/errs[1:])/np.log(hvals[:-1]/hvals[1:])
    print('estimated order: max = ', np.max(orders), ',  avg = ', np.average(orders))

    # forward Euler tests
    print(colored("\nforward Euler tests:", "yellow", attrs=["bold"]))
    for idx, h in enumerate(hvals):
        print("  h = ",h,",  lambda = ", lam,":", sep='')
        FE.reset()
        Y, success = FE.Evolve(tspan, y0, h, args=(lam,))
        Yerr = np.abs(Y-Ytrue)
        errs[idx] = np.linalg.norm(Yerr,np.inf)
        if (success):
            print("     " + colored("  t      y(t)     |err(t)| ", attrs=["underline"]))
            for i in range(Nout):
                text = "      %.1f  %10.2e  %.2e" % (tspan[i], Y[i,0], Yerr[i,0])
                if (Yerr[i,0] > 1):
                    print(colored(text, "light_red"))
                else:
                    print(text)
            text = "  overall:  steps = %4i  abserr = %8.2e" % (FE.get_num_steps(), errs[idx])
            if (errs[idx] > 1):
                print(colored(text, "red"))
            else:
                print(colored(text, "green"))
    orders = np.log(errs[:-1]/errs[1:])/np.log(hvals[:-1]/hvals[1:])
    print('estimated order: max = ', np.max(orders), ',  avg = ', np.average(orders))
