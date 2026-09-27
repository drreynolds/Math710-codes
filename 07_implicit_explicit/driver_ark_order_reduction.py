#!/usr/bin/env python
#
# Main routine to demonstrate order reduction of fixed-step ARK methods on
# the split Prothero--Robinson problem
#
#   y' = fE(t,y) + fI(t,y),
#   fE(t,y) = mu*(y-phi(t)),                  (nonstiff, treated explicitly)
#   fI(t,y) = lambda*(y-phi(t)) + phi'(t),    (stiff, treated implicitly)
#   phi(t) = sin(t+pi/4),  mu = -1,  t in [0,10],
#
# that has analytical solution y(t) = phi(t).  Since fE vanishes on the true
# solution, the explicit table does not introduce stage errors of its own, so
# the stiff convergence rates are governed by the stage order of the implicit
# table.  The ARS(3,4,3) implicit table has stage order one, whereas the
# implicit table in ARK3(2)4L[2]SA has stage order two.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from shared.ImplicitSolver import *
from ARK import *

# problem time interval and parameters
t0 = 0.0
tf = 10.0
mu = -1.0

# problem-defining functions
def ytrue(t):
    """ Generates a numpy array containing the true solution to the IVP. """
    return np.array([np.sin(t + np.pi/4.0)])
def fE(t,y,lam):
    """ Explicit portion of the split Prothero--Robinson problem. """
    return np.array([mu*(y[0] - ytrue(t)[0])])
def fI(t,y,lam):
    """ Implicit portion of the split Prothero--Robinson problem. """
    return np.array([lam*(y[0] - ytrue(t)[0]) + np.cos(t + np.pi/4.0)])
def JI(t,y,lam):
    """ Jacobian of the implicit portion of the right-hand side. """
    return np.array([[lam]])

# shared testing data
y0 = ytrue(t0)
tspan = np.array([t0, tf])
lambdas = -10.0**np.arange(1,5)
hvals = tf / (10 * 2**np.arange(0,11,2))

# short utility function to add log-log reference lines
def AddTrendLines(ax, order):
    """ Add short log-log reference lines for slopes 1 through order. """
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
def RunTest(BE, BI, name, stage_order):

    print("\n", name, " tests:", sep='')
    fig, ax = plt.subplots(figsize=(6.0,5.5))
    solver = ImplicitSolver(JI, solver_type='dense', maxiter=8,
                            rtol=1e-12, atol=1e-14, Jfreq=1)
    stepper = ARK(fE, fI, solver, BE, BI)
    for ilam, lam in enumerate(lambdas):
        errs = np.zeros(hvals.size)
        print("  lambda = ", lam, ":", sep='')
        for idx, h in enumerate(hvals):
            print("    h = %.5e:" % (h), sep='', end='')
            stepper.reset()
            stepper.sol.reset()
            Y, success = stepper.Evolve(tspan, y0, h, args=(lam,))
            errs[idx] = np.linalg.norm(Y[-1,:] - ytrue(tf), np.inf)
            if (success):
                print("  solves = %5i  Niters = %6i  NJevals = %5i  abserr = %8.2e" %
                      (stepper.get_num_solves(), stepper.sol.get_total_iters(),
                       stepper.sol.get_total_setups(), errs[idx]))
            else:
                print("  solve failed  abserr = %8.2e" % (errs[idx]))
        ax.loglog(hvals, errs, '-o', markersize=4,
                  label=r'$\lambda=-10^{%i}$' % (ilam+1))

    AddTrendLines(ax, 3)
    ax.set_xlabel(r'$h$')
    ax.set_ylabel('error')
    ax.set_title('%s (order 3, stage order %i)' % (name, stage_order))
    ax.grid(True, which='major', linestyle=':', linewidth=0.7)
    ax.legend(loc='best')
    fig.tight_layout()
    filename = 'ark_order_reduction_' + name.replace('(','').replace(')','').replace(',','') + '.png'
    fig.savefig(filename)
    print("  saved", filename)


BE, BI = ARS343()
RunTest(BE, BI, 'ARS(3,4,3)', 1)
BE, BI = ARK324L2SA()
RunTest(BE, BI, 'ARK324L2SA', 2)

plt.show()

# end of script
