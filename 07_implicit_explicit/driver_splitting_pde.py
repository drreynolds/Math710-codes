#!/usr/bin/env python
#
# Main routine to apply fractional-step (operator-splitting) methods to the
# viscous Burgers problem from shared/AdvectionDiffusion.py,
#    u' = fE(t,u) + fI(t,u),
# where fE is the (nonstiff) advection term and fI is the (stiff) diffusion
# and forcing term.  Each partition is advanced by its own sub-solver:
#
#   fixed:     advection with the fixed-step ERK method, using a step at the
#              explicit stability limit hE (so several sub-steps per fractional
#              step), and diffusion with one step of the fixed-step DIRK
#              method per fractional step;
#   adaptive:  advection with AdaptERK, and diffusion with AdaptDIRK, both at
#              a tight tolerance, so that the remaining error is essentially
#              the splitting error.
#
# Both use the explicit and implicit component tables of ARK3(2)4L[2]SA, so
# that the two sub-solver types can be compared directly.  We only use
# splittings with nonnegative sub-steps, since a backward-in-time sub-step of
# the diffusion term would be unstable.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import time
import sys
sys.path.append('..')
from shared.ImplicitSolver import *
from shared import AdvectionDiffusion as ad
sys.path.append('../04_explicit_one_step')
from ERK import ERK
from AdaptERK import AdaptERK
sys.path.append('../05_implicit_one_step')
from DIRK import DIRK
from AdaptDIRK import AdaptDIRK
from ARK import ARK324L2SA
from FractionalStep import *

# fractional-step methods to test, from the catalogue in FractionalStep.py
methods = [('Lie-Trotter', LieTrotter()),
           ('Strang-Marchuk', StrangMarchuk()),
           ('OS2(2,2)-1/4', OS2(0.25))]

# testing data
Hvals = 0.1 / 2.0**np.arange(5)
rtol = 1e-8
atol = 1e-12
hE = ad.explicit_stability_limit()
BE, BI = ARK324L2SA()
tspan = np.array([ad.t0, ad.tf])
y0 = ad.u0()
ytrue = ad.utrue(ad.xgrid, ad.tf)

# utility routines to construct each type of sub-solver pair
def FixedSolvers():
    solverI = ImplicitSolver(ad.JI, solver_type='sparse', maxiter=12,
                             rtol=1e-10, atol=1e-12, Jfreq=3)
    return [ERK(ad.fE, BE, hE), DIRK(ad.fI, solverI, BI, ad.tf-ad.t0)]
def AdaptiveSolvers():
    solverI = ImplicitSolver(ad.JI, solver_type='sparse', maxiter=12,
                             rtol=1e-10, atol=1e-12, Jfreq=3)
    return [AdaptERK(ad.fE, y0, BE, rtol=rtol, atol=atol),
            AdaptDIRK(ad.fI, y0, solverI, BI, rtol=rtol, atol=atol)]

# test runner function
def RunTest(name, S, BuildSolvers):
    print("\n  %s (splitting order %i):" % (name, S['p']))
    errs = np.zeros(Hvals.size)
    for idx, H in enumerate(Hvals):
        solvers = BuildSolvers()
        stepper = FractionalStep(S, solvers)
        tstart = time.perf_counter()
        Y, success = stepper.Evolve(tspan, y0, H)
        runtime = time.perf_counter() - tstart
        errs[idx] = np.max(np.abs(Y[-1,:] - ytrue))
        if (not success):
            print("    H = %.5f:  solve failed" % (H))
            continue
        print("    H = %.5f:  advection steps = %6i  diffusion steps = %5i  abserr = %8.2e  runtime = %.2fs" %
              (H, solvers[0].get_num_steps(), solvers[1].get_num_steps(), errs[idx], runtime))
    orders = np.log(errs[0:-1]/errs[1:])/np.log(Hvals[0:-1]/Hvals[1:])
    print("    estimated orders:  %s" % (' '.join('%5.2f' % q for q in orders)))

print("Burgers splitting tests, fixed sub-solvers (hE = %.2e for advection, one DIRK step for diffusion):" % (hE))
for name, S in methods:
    RunTest(name, S, FixedSolvers)

print("\nBurgers splitting tests, adaptive sub-solvers (rtol = %.0e):" % (rtol))
for name, S in methods:
    RunTest(name, S, AdaptiveSolvers)

# end of script
