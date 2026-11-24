#!/usr/bin/env python
#
# Main routine to examine the steady states of fractional-step
# (operator-splitting) methods.  We consider the scalar problem
#    y' = -y + 1 = f1(y) + f2(y),   t in [0, Tf],   y(0) = 0,
# with the "decay" piece f1(y) = -y and the "source" piece f2(y) = 1.  Its
# exact solution y(t) = 1 - exp(-t) approaches the steady state y* = 1, at
# which the two pieces balance.  Neither piece is zero there, however, so
# each sub-step of a splitting method moves the solution away from y* = 1,
# and the method instead converges to the fixed point of its own one-step map.
# With exact sub-flows these fixed points are:
#    Lie-Trotter (decay, then source):      H / (1 - exp(-H))   ~ 1 + H/2
#    Lie-Trotter adjoint (source, decay):   H / (exp(H) - 1)    ~ 1 - H/2
#    Strang (half steps of decay):          (H/2) / sinh(H/2)   ~ 1 - H^2/24
#    Strang (half steps of source):         (H/2) coth(H/2)     ~ 1 + H^2/12
#
# We approximate the exact sub-flows by taking many RK4 sub-steps per
# fractional step (the source piece is integrated exactly by any Runge--Kutta
# method, and the RK4 error for the decay piece is negligible here).  The
# script runs each method to Tf = 30, long enough that every computed
# solution has reached its fixed point, prints the steady-state error
# y(Tf) - 1 alongside the predicted value, and estimates the order of the
# steady-state error in H.  It then plots the Lie-Trotter and Strang
# solutions for several step sizes, to show that each levels off at its own,
# incorrect, steady state.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../04_explicit_one_step')
from ERK import *
from FractionalStep import *

# problem time interval and initial condition
t0 = 0.0
tf = 30.0
y0 = np.array([0.0])
tspan = np.array([t0, tf])

# exact solution and steady state
def ytrue(t):
    return 1.0 - np.exp(-t)
ystar = 1.0

# right-hand side pieces
def fdecay(t, y):
    """ Decay piece of the right-hand side """
    return -y
def fsource(t, y):
    """ Source piece of the right-hand side """
    return np.ones_like(y)

# RK4 sub-integrator step size; the sub-flows are then accurate to roughly
# 1e-10, far below the steady-state errors that we wish to observe
hsub = 1e-2

# fractional-step methods to test: (name, coefficient table, ordering of the
# right-hand side pieces, predicted fixed point as a function of H).  The
# fourth method is Strang--Marchuk splitting with the partitions exchanged,
# so that the source piece takes the two half steps.
methods = [('Lie-Trotter', LieTrotter(), [fdecay, fsource],
            lambda H: H/(1.0 - np.exp(-H))),
           ('Lie-Trotter adjoint', LieTrotterAdjoint(), [fdecay, fsource],
            lambda H: H/(np.exp(H) - 1.0)),
           ('Strang (decay halves)', StrangMarchuk(), [fdecay, fsource],
            lambda H: (H/2)/np.sinh(H/2)),
           ('Strang (source halves)', StrangMarchuk(), [fsource, fdecay],
            lambda H: (H/2)/np.tanh(H/2))]

# step sizes to test
Hvals = 1.0 / 2.0**np.arange(5)

# test runner function: returns the solution at the times in tspan
def RunSplitting(S, f, H, hsub, tspan, y0):
    """
    Runs the fractional-step method with coefficient table S, using many RK4
    sub-steps of size hsub for each partition l, with right-hand side
    functions f[l], outer step size H, and initial condition y0.
    """
    solvers = [ERK(f[l], ERK4(), hsub) for l in range(len(f))]
    stepper = FractionalStep(S, solvers)
    Y, success = stepper.Evolve(tspan, y0, H)
    return Y[:,0]


#### Steady-state errors ####
print("Steady-state errors of splitting methods for y' = -y + 1 (true steady state y* = 1)")
for name, S, f, fixed_point in methods:
    print("\n  %s (splitting order %i):" % (name, S['p']))
    errs = np.zeros(Hvals.size)
    for idx, H in enumerate(Hvals):
        Y = RunSplitting(S, f, H, hsub, tspan, y0)
        errs[idx] = Y[-1] - ystar
        if (idx == 0):
            print("    H = %6.4f:  y(Tf) - 1 = %10.3e,  predicted = %10.3e" %
                  (H, errs[idx], fixed_point(H) - ystar))
        else:
            order = np.log(abs(errs[idx-1]/errs[idx]))/np.log(Hvals[idx-1]/Hvals[idx])
            print("    H = %6.4f:  y(Tf) - 1 = %10.3e,  predicted = %10.3e,  order = %5.2f" %
                  (H, errs[idx], fixed_point(H) - ystar, order))


#### Solution plots ####
# output at every outer step, so that each plotted point is a computed solution
Hplot = [1.0, 0.5, 0.25]
tfine = np.linspace(t0, 8.0, 401)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, S, f, fixed_point) in zip(axs, [methods[0], methods[2]]):
    ax.plot(tfine, ytrue(tfine), 'k-', label='exact')
    for H in Hplot:
        tout = t0 + H*np.arange(int(round(8.0/H)) + 1)
        Y = RunSplitting(S, f, H, hsub, tout, y0)
        ax.plot(tout, Y, 'o-', markersize=4, label='H = %g' % (H))
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.set_title(name)
    ax.grid(True, linestyle=':')
    ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('splitting_steady_state.png')
plt.show()
