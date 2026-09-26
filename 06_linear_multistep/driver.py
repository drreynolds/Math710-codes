#!/usr/bin/env python
#
# Main routine to test various DIRK and IRK methods on the
# scalar-valued ODE problem
#    y' = lambda*y + (1-lambda)*cos(t) - (1+lambda)*sin(t), t in [0,5],
#    y(0) = 1.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import sys
sys.path.append('..')
from shared.ImplicitSolver import *
from Explicit_LMM import *
from Implicit_LMM import *

# problem time interval and parameters
t0 = 0.0
tf = 5.0

# flag to switch between dense and iterative linear solvers
iterative = False

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
def Jv(t,y,v,lam):
    """ Jacobian (in dense matrix format) of the right-hand side function, J(t,y) = df/dy """
    return np.array( [lam*v[0]] )

# construct implicit solver
if (iterative):
    solver = ImplicitSolver(Jv, solver_type='gmres', maxiter=20, rtol=1e-9, atol=1e-12)
else:
    solver = ImplicitSolver(J, solver_type='dense', maxiter=20, rtol=1e-9, atol=1e-12, Jfreq=2)

# shared testing data
Nout = 6   # includes initial condition
tspan = np.linspace(t0, tf, Nout)
Ytrue = np.zeros((Nout, 1))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])
lambdas = np.array( (-1.0, -10.0, -50.0, -100.0) )
hvals = np.array( (0.1, 0.05, 0.01, 0.005, 0.001) )
errs = np.zeros(hvals.size)

# test runner function
def RunTest(stepper, prevsteps, name, implicit):

    print("\n", name, " tests:", sep='')
    # loop over stiffness values
    for lam in lambdas:

        print("  lambda = " , lam, ":", sep='')
        for idx, h in enumerate(hvals):
            print("    h = %.3f:" % (h), sep='', end='')
            stepper.reset()
            if (implicit):
                stepper.sol.reset()
            # create overly-long initial condition vector (sufficient for all methods)
            y0 = np.zeros((prevsteps+1, ytrue(t0).size), dtype=float)
            y0[-1,:] = ytrue(t0)
            for k in range(1,prevsteps+1):
                y0[-1-k,:] = ytrue(t0-k*h)
            # Note that this is where we provide the rhs function parameter lam -- the "," is
            # required to ensure that args is an iterable (and not a float).
            Y, success = stepper.Evolve(tspan, y0, h, args=(lam,))
            Yerr = np.abs(Y-Ytrue)
            errs[idx] = np.linalg.norm(Yerr,np.inf)
            if (success):
                if (implicit):
                    print("  steps = %4i  Niters = %6i  NJevals = %5i  abserr = %8.2e" %
                          (stepper.get_num_steps(), stepper.sol.get_total_iters(),
                           stepper.sol.get_total_setups(), errs[idx]))
                else:
                    print("  steps = %4i  Nrhs = %6i  abserr = %8.2e" %
                          (stepper.get_num_steps(), stepper.get_num_rhs(), errs[idx]))
            else:
                print("  solve failed  abserr = %8.2e" % (errs[idx]))
        if (hvals.size > 2):
            orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
            print('    estimated order:  max = %.2f,  avg = %.2f' %
                  (np.max(orders), np.average(orders)))


# Adams-Bashforth-1
alphas, betas, p = AdamsBashforth1()
AB1 = Explicit_LMM(f, alphas, betas)
RunTest(AB1, 0, 'Adams-Bashforth-1', False)

# Adams-Bashforth-2
alphas, betas, p = AdamsBashforth2()
AB2 = Explicit_LMM(f, alphas, betas)
RunTest(AB2, 1, 'Adams-Bashforth-2', False)

# Adams-Bashforth-3
alphas, betas, p = AdamsBashforth3()
AB3 = Explicit_LMM(f, alphas, betas)
RunTest(AB3, 2, 'Adams-Bashforth-3', False)

# Adams-Bashforth-4
alphas, betas, p = AdamsBashforth4()
AB4 = Explicit_LMM(f, alphas, betas)
RunTest(AB4, 3, 'Adams-Bashforth-4', False)

# Adams-Moulton-1
alphas, betas, p = AdamsMoulton1()
AM1 = Implicit_LMM(f, solver, alphas, betas)
RunTest(AM1, 0, 'Adams-Moulton-1', True)

# Adams-Moulton-2
alphas, betas, p = AdamsMoulton2()
AM2 = Implicit_LMM(f, solver, alphas, betas)
RunTest(AM2, 0, 'Adams-Moulton-2', True)

# Adams-Moulton-3
alphas, betas, p = AdamsMoulton3()
AM3 = Implicit_LMM(f, solver, alphas, betas)
RunTest(AM3, 1, 'Adams-Moulton-3', True)

# Adams-Moulton-4
alphas, betas, p = AdamsMoulton4()
AM4 = Implicit_LMM(f, solver, alphas, betas)
RunTest(AM4, 2, 'Adams-Moulton-4', True)

# BDF-1
alphas, betas, p = BDF1()
BDF_1 = Implicit_LMM(f, solver, alphas, betas)
RunTest(BDF_1, 0, 'BDF-1', True)

# BDF-2
alphas, betas, p = BDF2()
BDF_2 = Implicit_LMM(f, solver, alphas, betas)
RunTest(BDF_2, 1, 'BDF-2', True)

# BDF-3
alphas, betas, p = BDF3()
BDF_3 = Implicit_LMM(f, solver, alphas, betas)
RunTest(BDF_3, 2, 'BDF-3', True)

# BDF-4
alphas, betas, p = BDF4()
BDF_4 = Implicit_LMM(f, solver, alphas, betas)
RunTest(BDF_4, 3, 'BDF-4', True)
