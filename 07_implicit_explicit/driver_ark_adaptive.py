#!/usr/bin/env python
#
# Main routine to compare adaptive ARK, DIRK, and ERK methods on the
# reaction-diffusion and viscous Burgers problems.  The ARK solver treats
# diffusion implicitly and all remaining terms explicitly.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from shared.ImplicitSolver import *
from shared import ReactionDiffusion as rd
from shared import AdvectionDiffusion as ad
sys.path.append('../04_explicit_one_step')
from AdaptERK import AdaptERK
sys.path.append('../05_implicit_one_step')
from AdaptDIRK import AdaptDIRK
from AdaptARK import *

# shared testing data
rtols = 10.0**(-np.arange(3,8,2))
atol = 1.e-12
Nout = 20

# test runner function
def RunProblem(problem, problem_name, hE=np.inf):

    print("\n", problem_name, " tests:", sep='')
    tspan = np.linspace(problem.t0, problem.tf, Nout+1)
    ytrue = np.zeros((Nout+1, problem.xgrid.size), dtype=float)
    for iout in range(Nout+1):
        ytrue[iout,:] = problem.utrue(problem.xgrid, tspan[iout])
    y0 = ytrue[0,:]

    # use the same component tables for all three solvers
    BE, BI = ARK324L2SA()
    ark_iters = np.zeros(rtols.size, dtype=int)
    ark_times = np.zeros(rtols.size)
    ark_errs = np.zeros(rtols.size)
    dirk_iters = np.zeros(rtols.size, dtype=int)
    dirk_times = np.zeros(rtols.size)
    dirk_errs = np.zeros(rtols.size)
    erk_times = np.zeros(rtols.size)
    erk_errs = np.zeros(rtols.size)
    ark_history = None

    for idx, rtol in enumerate(rtols):
        print("  rtol = %.1e:" % (rtol))

        # adaptive ARK method on the split problem
        solverI = ImplicitSolver(problem.JI, solver_type='sparse', maxiter=12,
                                 rtol=1e-10, atol=1e-12, Jfreq=3)
        A = AdaptARK(problem.fE, problem.fI, y0, solverI, BE, BI,
                     rtol=rtol, atol=atol, hE=hE, save_step_hist=True)
        tstart = time.perf_counter()
        Y, success = A.Evolve(tspan, y0)
        ark_times[idx] = time.perf_counter() - tstart
        ark_iters[idx] = solverI.get_total_iters()
        ark_errs[idx] = np.max(np.abs(Y-ytrue))
        ark_history = A.get_step_history()
        if (success):
            print("    ARK:   steps = %6i  fails = %3i  solves = %6i  Niters = %7i  NJevals = %6i  abserr = %8.2e  runtime = %.2fs" %
                  (A.get_num_steps(), A.get_num_error_failures(), A.get_num_solves(),
                   solverI.get_total_iters(), solverI.get_total_setups(), ark_errs[idx], ark_times[idx]))
        else:
            print("    ARK:   solve failed  abserr = %8.2e" % (ark_errs[idx]))

        # adaptive DIRK method on the full right-hand side
        solver = ImplicitSolver(problem.J, solver_type='sparse', maxiter=12,
                                rtol=1e-10, atol=1e-12, Jfreq=3)
        D = AdaptDIRK(problem.f, y0, solver, BI, rtol=rtol, atol=atol)
        tstart = time.perf_counter()
        Y, success = D.Evolve(tspan, y0)
        dirk_times[idx] = time.perf_counter() - tstart
        dirk_iters[idx] = solver.get_total_iters()
        dirk_errs[idx] = np.max(np.abs(Y-ytrue))
        if (success):
            print("    DIRK:  steps = %6i  fails = %3i  solves = %6i  Niters = %7i  NJevals = %6i  abserr = %8.2e  runtime = %.2fs" %
                  (D.get_num_steps(), D.get_num_error_failures(), D.get_num_solves(),
                   solver.get_total_iters(), solver.get_total_setups(), dirk_errs[idx], dirk_times[idx]))
        else:
            print("    DIRK:  solve failed  abserr = %8.2e" % (dirk_errs[idx]))

        # adaptive ERK method on the full right-hand side
        E = AdaptERK(problem.f, y0, BE, rtol=rtol, atol=atol)
        tstart = time.perf_counter()
        Y, success = E.Evolve(tspan, y0)
        erk_times[idx] = time.perf_counter() - tstart
        erk_errs[idx] = np.max(np.abs(Y-ytrue))
        if (success):
            print("    ERK:   steps = %6i  fails = %3i  nrhs = %7i  abserr = %8.2e  runtime = %.2fs" %
                  (E.get_num_steps(), E.get_num_error_failures(), E.get_num_rhs(), erk_errs[idx], erk_times[idx]))
        else:
            print("    ERK:   solve failed  abserr = %8.2e" % (erk_errs[idx]))

    # accuracy-versus-Newton-iterations plot
    plt.figure()
    plt.loglog(ark_iters, ark_errs, 'bo-', label='ARK3(2)4L[2]SA')
    plt.loglog(dirk_iters, dirk_errs, 'rs-', label='DIRK component on full RHS')
    plt.xlabel('Newton iterations')
    plt.ylabel('error')
    plt.title(problem_name + ' -- adaptive accuracy versus Newton iterations')
    plt.grid(True, which='major', linestyle=':', linewidth=0.7)
    plt.legend()
    plt.savefig('ark_adaptive_' + problem_name.lower().replace('-', '_') + '_iters.png')

    # accuracy-versus-runtime plot
    plt.figure()
    plt.loglog(ark_times, ark_errs, 'bo-', label='ARK3(2)4L[2]SA')
    plt.loglog(dirk_times, dirk_errs, 'rs-', label='DIRK component on full RHS')
    plt.loglog(erk_times, erk_errs, 'g^-', label='ERK component on full RHS')
    plt.xlabel('runtime (s)')
    plt.ylabel('error')
    plt.title(problem_name + ' -- adaptive accuracy versus runtime')
    plt.grid(True, which='major', linestyle=':', linewidth=0.7)
    plt.legend()
    plt.savefig('ark_adaptive_' + problem_name.lower().replace('-', '_') + '_runtime.png')

    # ARK step history at the tightest tolerance
    plt.figure()
    plt.plot(ark_history['t'], ark_history['h'], 'b-')
    for i in range(len(ark_history['t'])):
        if (ark_history['err'][i] > 1.0):
            plt.plot(ark_history['t'][i], ark_history['h'][i], 'bx')
    if (np.isfinite(hE)):
        plt.plot([problem.t0, problem.tf], [hE, hE], 'k--', label=r'$h_E$')
        plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$h$')
    plt.title(problem_name + ' -- adaptive ARK step history')
    plt.savefig('ark_adaptive_' + problem_name.lower().replace('-', '_') + '_steps.png')


RunProblem(rd, 'Reaction-diffusion')
RunProblem(ad, 'Burgers', hE=ad.explicit_stability_limit())

plt.show()

# end of script
