#!/usr/bin/env python
#
# Main routine to demonstrate order reduction of fixed-step DIRK methods on
# the linear Prothero--Robinson ODE from Ketcheson, Seibold, Shirokoff, and
# Zhou (2020), Sect. 4.1:
#
#   u' = lambda*(u - phi(t)) + phi'(t),  phi(t) = sin(t + pi/4),  t in [0,10].
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../shared')
from ImplicitSolver import *
from DIRK import *

# problem time interval and parameters
t0 = 0.0
tf = 10.0

# problem-defining functions
def ytrue(t):
    """Generates a numpy array containing the true solution to the IVP."""
    return np.array([np.sin(t + np.pi/4.0)])

def f(t, y, lam):
    """Right-hand side function for the linear Prothero--Robinson problem."""
    return np.array([lam*(y[0] - ytrue(t)[0]) + np.cos(t + np.pi/4.0)])

def J(t, y, lam):
    """Jacobian of the right-hand side function."""
    return np.array([[lam]])

# shared testing data.  Each h is tf/N for an integer N, so the DIRK solver
# uses exactly the h value plotted below.
y0 = ytrue(t0)
tspan = np.array([t0, tf])
lambdas = -10.0**np.arange(1, 5)
hvals = tf / (10 * 2**np.arange(0, 11, 2))

# short utility function to add log-log reference lines for slopes 1 through order.
def AddTrendLines(ax, order):
    """Add short log-log reference lines for slopes 1 through order."""
    htrend = np.array([hvals[-1], hvals[-3]])
    hratio = htrend[1] / htrend[0]
    ymin, ymax = ax.get_ylim()
    logymin = np.log10(ymin)
    logymax = np.log10(ymax)
    upper_start = logymax - order*np.log10(hratio)
    logystarts = np.linspace(logymin + 0.1, upper_start - 0.1, order)
    trend_colors = plt.cm.Dark2(np.linspace(0.0, 1.0, order))

    for slope, (logystart, color) in enumerate(zip(logystarts, trend_colors), start=1):
        ytrend = 10.0**logystart * (htrend / htrend[0])**slope
        ax.loglog(htrend, ytrend, '--', color=color, linewidth=1.5,
                  label='slope %i' % slope)

# test runner function
def RunTest(stepper, name, order, wso):

    print("\n", name, " tests:", sep='')
    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    # loop over stiffness values
    for ilam, lam in enumerate(lambdas):

        errs = np.zeros(hvals.size)
        print("  lambda = ", lam, ":", sep='')
        for idx, h in enumerate(hvals):
            print("    h = %.5e:" % (h), sep='', end='')
            stepper.reset()
            stepper.sol.reset()
            # The comma is required so that args is an iterable, not a float.
            Y, success = stepper.Evolve(tspan, y0, h, args=(lam,))
            if (not success):
                raise RuntimeError("DIRK solve failed for lambda=%g, h=%g" % (lam, h))
            errs[idx] = np.linalg.norm(Y[-1,:] - ytrue(tf), np.inf)
            print("  solves = %5i  Niters = %6i  NJevals = %5i  abserr = %8.2e" %
                  (stepper.get_num_solves(), stepper.sol.get_total_iters(),
                   stepper.sol.get_total_setups(), errs[idx]))

        ax.loglog(hvals, errs, '-o', markersize=4,
                  label=r'$\lambda=-10^{%i}$' % (ilam+1))

    AddTrendLines(ax, order)
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'error')
    ax.set_title('%s (order %i, WSO %i)' % (name, order, wso))
    ax.grid(True, which='major', linestyle=':', linewidth=0.7)
    ax.legend(loc='best')
    fig.tight_layout()
    filename = 'order_reduction_%s.png' % name
    fig.savefig(filename)
    print("  saved", filename)


# Shared nonlinear solver; RunTest resets its statistics before each solve.
solver = ImplicitSolver(J, solver_type='dense', maxiter=8,
                        rtol=1e-12, atol=1e-14, Jfreq=1)

# The first three are conventional order >= 3 DIRK methods; the final three
# are the high-WSO methods published in Sect. 3 of the paper.
Alex3 = DIRK(f, solver, Alexander3())
SD45 = DIRK(f, solver, SDIRK45L1SA())
C6 = DIRK(f, solver, Cooper6ESDIRK())
D32 = DIRK(f, solver, WSO32())
D33 = DIRK(f, solver, WSO33())
D43 = DIRK(f, solver, WSO43())

RunTest(Alex3, 'Alexander3', 3, 1)
RunTest(SD45, 'SDIRK45L1SA', 4, 1)
RunTest(C6, 'Cooper6ESDIRK', 5, 1)
RunTest(D32, 'WSO32', 3, 2)
RunTest(D33, 'WSO33', 3, 3)
RunTest(D43, 'WSO43', 4, 3)

plt.show()
