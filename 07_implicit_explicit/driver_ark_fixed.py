#!/usr/bin/env python
#
# Main routine to test fixed-step ARK methods on the reaction-diffusion
# problem from shared/ReactionDiffusion.py.  Diffusion is treated implicitly,
# while the reaction and forcing terms are treated explicitly.  Results are
# compared against a DIRK method applied to the full right-hand side.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import sys
sys.path.append('..')
from shared.ImplicitSolver import *
from shared import ReactionDiffusion as rd
sys.path.append('../05_implicit_one_step')
from DIRK import *
from ARK import *

# shared testing data
Nout = 1
tspan = np.linspace(rd.t0, rd.tf, Nout+1)
ytrue = np.zeros((Nout+1, rd.Nx-2), dtype=float)
for iout in range(Nout+1):
    ytrue[iout,:] = rd.utrue(rd.xgrid, tspan[iout])
y0 = ytrue[0,:]
hvals = 0.01 / 2.0**np.arange(4)
errs = np.zeros(hvals.size)

# test runner function for ARK methods
def RunARKTest(BE, BI, name):

    print("\n", name, " tests:", sep='')
    solver = ImplicitSolver(rd.JI, solver_type='sparse', maxiter=20,
                            rtol=1e-10, atol=1e-12, Jfreq=3)
    stepper = ARK(rd.fE, rd.fI, solver, BE, BI)
    for idx, h in enumerate(hvals):
        print("  h = %.5e:" % (h), sep='', end='')
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
    print('  estimated order:  max = %.2f,  avg = %.2f' %
          (np.max(orders), np.average(orders)))

# test runner function for the full-RHS DIRK method
def RunDIRKTest(B, name):

    print("\n", name, " tests:", sep='')
    solver = ImplicitSolver(rd.J, solver_type='sparse', maxiter=20,
                            rtol=1e-10, atol=1e-12, Jfreq=3)
    stepper = DIRK(rd.f, solver, B)
    for idx, h in enumerate(hvals):
        print("  h = %.5e:" % (h), sep='', end='')
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
    print('  estimated order:  max = %.2f,  avg = %.2f' %
          (np.max(orders), np.average(orders)))

# fixed-step ARK tests
BE, BI = ARS122()
RunARKTest(BE, BI, 'ARS(1,2,2)')
BE, BI = ARS343()
RunARKTest(BE, BI, 'ARS(3,4,3)')
BE, BI = ARK324L2SA()
RunARKTest(BE, BI, 'ARK3(2)4L[2]SA')
BE, BI = ARK436L2SA()
RunARKTest(BE, BI, 'ARK4(3)6L[2]SA')

# full-RHS DIRK comparison
RunDIRKTest(CrouzeixRaviart3(), 'Crouzeix-Raviart-3 on full RHS')

# end of script
