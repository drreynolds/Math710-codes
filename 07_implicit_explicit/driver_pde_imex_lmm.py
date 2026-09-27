#!/usr/bin/env python
#
# Main routine to test fixed-step ImEx LMM methods on the reaction-diffusion
# problem from shared/ReactionDiffusion.py and the viscous Burgers problem from
# shared/AdvectionDiffusion.py.  In both, the diffusion and forcing terms are
# treated implicitly, while the reaction or advection term is treated
# explicitly.  Results are compared against a second-order DIRK method applied
# to the full right-hand side.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import sys
sys.path.append('..')
from shared.ImplicitSolver import *
from shared import ReactionDiffusion as rd
from shared import AdvectionDiffusion as ad
sys.path.append('../05_implicit_one_step')
from DIRK import *
from ImEx_LMM import *

# shared testing data
Nout = 1
hvals = 0.01 / 2.0**np.arange(4)
errs = np.zeros(hvals.size)

# test runner function for ImEx LMM methods
def RunLMMTest(problem, alphas, betas, gammas, prevsteps, name, tspan, ytrue):

    print("\n  ", name, " tests:", sep='')
    solver = ImplicitSolver(problem.JI, solver_type='sparse', maxiter=20,
                            rtol=1e-10, atol=1e-12, Jfreq=3)
    stepper = ImEx_LMM(problem.fE, problem.fI, solver, alphas, betas, gammas)
    for idx, h in enumerate(hvals):
        print("    h = %.5e:" % (h), sep='', end='')
        stepper.reset()
        stepper.sol.reset()
        # create initial condition vector with the required previous solution values
        y0 = np.zeros((prevsteps+1, problem.xgrid.size), dtype=float)
        y0[-1,:] = ytrue[0,:]
        for k in range(1,prevsteps+1):
            y0[-1-k,:] = problem.utrue(problem.xgrid, tspan[0]-k*h)
        Y, success = stepper.Evolve(tspan, y0, h)
        errs[idx] = np.max(np.abs(Y-ytrue))
        if (success):
            print("  steps = %4i  solves = %5i  Niters = %6i  NJevals = %5i  abserr = %8.2e" %
                  (stepper.get_num_steps(), stepper.get_num_solves(),
                   stepper.sol.get_total_iters(), stepper.sol.get_total_setups(), errs[idx]))
        else:
            print("  solve failed  abserr = %8.2e" % (errs[idx]))
    orders = np.log(errs[0:-1]/errs[1:])/np.log(hvals[0:-1]/hvals[1:])
    print('    estimated order:  max = %.2f,  avg = %.2f' %
          (np.max(orders), np.average(orders)))

# test runner function for the full-RHS DIRK method
def RunDIRKTest(problem, B, name, tspan, ytrue):

    print("\n  ", name, " tests:", sep='')
    solver = ImplicitSolver(problem.J, solver_type='sparse', maxiter=20,
                            rtol=1e-10, atol=1e-12, Jfreq=3)
    stepper = DIRK(problem.f, solver, B)
    y0 = ytrue[0,:]
    for idx, h in enumerate(hvals):
        print("    h = %.5e:" % (h), sep='', end='')
        stepper.reset()
        stepper.sol.reset()
        Y, success = stepper.Evolve(tspan, y0, h)
        errs[idx] = np.max(np.abs(Y-ytrue))
        if (success):
            print("  steps = %4i  solves = %5i  Niters = %6i  NJevals = %5i  abserr = %8.2e" %
                  (stepper.get_num_steps(), stepper.get_num_solves(),
                   stepper.sol.get_total_iters(), stepper.sol.get_total_setups(), errs[idx]))
        else:
            print("  solve failed  abserr = %8.2e" % (errs[idx]))
    orders = np.log(errs[0:-1]/errs[1:])/np.log(hvals[0:-1]/hvals[1:])
    print('    estimated order:  max = %.2f,  avg = %.2f' %
          (np.max(orders), np.average(orders)))

# problem runner function
def RunProblem(problem, problem_name):

    print("\n", problem_name, " tests:", sep='')
    tspan = np.linspace(problem.t0, problem.tf, Nout+1)
    ytrue = np.zeros((Nout+1, problem.xgrid.size), dtype=float)
    for iout in range(Nout+1):
        ytrue[iout,:] = problem.utrue(problem.xgrid, tspan[iout])

    # ImEx LMM tests
    alphas, betas, gammas, p = SBDF2()
    RunLMMTest(problem, alphas, betas, gammas, 1, 'SBDF-2', tspan, ytrue)
    alphas, betas, gammas, p = CNAB()
    RunLMMTest(problem, alphas, betas, gammas, 1, 'CNAB', tspan, ytrue)
    alphas, betas, gammas, p = MCNAB()
    RunLMMTest(problem, alphas, betas, gammas, 1, 'MCNAB', tspan, ytrue)
    alphas, betas, gammas, p = CNLF()
    RunLMMTest(problem, alphas, betas, gammas, 1, 'CNLF', tspan, ytrue)

    # full-RHS DIRK comparison
    RunDIRKTest(problem, SSP222SDIRK(), 'SSP2(2,2,2)-SDIRK on full RHS', tspan, ytrue)


RunProblem(rd, 'Reaction-diffusion')
RunProblem(ad, 'Burgers')

# end of script
