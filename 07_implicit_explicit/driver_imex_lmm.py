#!/usr/bin/env python
#
# Main routine to test various ImEx LMM methods on the
# scalar-valued additively split ODE problem
#    y' = fe(t,y) + fi(t,y), t in [0,2],
#    y(0) = 1,
# where
#    fe(t,y) = sin(y-cos(t)) + 0.3*(y-cos(t)) - 0.5*sin(t),
#    fi(t,y) = lambda*(y-cos(t)) - 0.5*sin(t),
# which has true solution y(t) = cos(t).
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import sys
sys.path.append('..')
from shared.ImplicitSolver import *
from ImEx_LMM import *

# problem time interval and parameters
t0 = 0.0
tf = 2.0

# problem-defining functions
def ytrue(t):
    """ Generates a numpy array containing the true solution to the IVP at a given input t. """
    return np.array([np.cos(t)])
def fe(t,y,lam):
    """ Explicit right-hand side function, fe(t,y), for the IVP """
    return np.array([np.sin(y[0] - np.cos(t)) + 0.3*(y[0] - np.cos(t)) - 0.5*np.sin(t)])
def fi(t,y,lam):
    """ Implicit right-hand side function, fi(t,y), for the IVP """
    return np.array([lam*(y[0] - np.cos(t)) - 0.5*np.sin(t)])
def J(t,y,lam):
    """ Jacobian (in dense matrix format) of the implicit right-hand side function, J(t,y) = dfi/dy """
    return np.array( [ [lam] ] )
def Jv(t,y,v,lam):
    """ Jacobian-vector product of the implicit right-hand side function, J(t,y)*v = dfi/dy*v """
    return np.array( [lam*v[0]] )

# construct implicit solver
solver = ImplicitSolver(J, solver_type='dense', maxiter=20, rtol=1e-12, atol=1e-14)

# shared testing data
Nout = 5   # includes initial condition
tspan = np.linspace(t0, tf, Nout)
Ytrue = np.zeros((Nout, 1))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])
lambdas = np.array( (-1.0, -1000.0) )
hvals = 0.01 / 2.0**np.arange(5)
errs = np.zeros(hvals.size)

# test runner function
def RunTest(stepper, prevsteps, name):

    print("\n", name, " tests:", sep='')
    # loop over stiffness values
    for lam in lambdas:

        print("  lambda = " , lam, ":", sep='')
        for idx, h in enumerate(hvals):
            print("    h = %.2e:" % (h), sep='', end='')
            stepper.reset()
            stepper.sol.reset()
            # create initial condition vector with the required previous solution values
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
                print("  steps = %4i  Niters = %6i  NJevals = %5i  abserr = %8.2e" %
                      (stepper.get_num_steps(), stepper.sol.get_total_iters(),
                       stepper.sol.get_total_setups(), errs[idx]))
            else:
                print("  solve failed  abserr = %8.2e" % (errs[idx]))
        if (hvals.size > 1):
            orders = np.log(errs[0:-1]/errs[1:])/np.log(hvals[0:-1]/hvals[1:])
            print('    estimated order:  max = %.2f,  avg = %.2f' %
                  (np.max(orders), np.average(orders)))


# SBDF-2
alphas, betas, gammas, p = SBDF2()
SBDF_2 = ImEx_LMM(fe, fi, solver, alphas, betas, gammas)
RunTest(SBDF_2, 1, 'SBDF-2')

# CNAB
alphas, betas, gammas, p = CNAB()
CNAB_ = ImEx_LMM(fe, fi, solver, alphas, betas, gammas)
RunTest(CNAB_, 1, 'CNAB')

# MCNAB
alphas, betas, gammas, p = MCNAB()
MCNAB_ = ImEx_LMM(fe, fi, solver, alphas, betas, gammas)
RunTest(MCNAB_, 1, 'MCNAB')

# CNLF
alphas, betas, gammas, p = CNLF()
CNLF_ = ImEx_LMM(fe, fi, solver, alphas, betas, gammas)
RunTest(CNLF_, 1, 'CNLF')
