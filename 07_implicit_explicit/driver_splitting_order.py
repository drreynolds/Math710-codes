#!/usr/bin/env python
#
# Main routine to examine the accuracy of fractional-step (operator-splitting)
# methods, using two experiments.
#
# Experiment 1: splitting order versus sub-integrator order.  We apply several
# two-way splittings to the nonlinear Kvaerno Prothero and Robinson problem:
#    [u]' = [ G  e ] [(-1+u^2-r)/(2u)] + [ r'(t)/(2u) ]
#    [v]    [ e -1 ] [(-2+v^2-s)/(2v)]   [ s'(t)/(2v) ]
#         = [f1(t,y)]
#           [f2(t,y)]
# where r(t) = 0.5*cos(t),  s(t) = cos(w*t),  0 < t < 5, which has analytical
# solution u(t) = sqrt(1+r(t)),  v(t) = sqrt(2+s(t)).  We split by components,
# with G = -1, e = 0.5 and w = 2, so that the two pieces evolve on similar
# time scales (this is not a multirate test), applying each splitting
# first with highly accurate sub-integrators (many RK4 sub-steps per
# fractional step), and then with one step of forward Euler, Heun, or RK3 per
# fractional step, to show that the observed order is the minimum of the
# splitting order and the sub-integrator order.
#
# Experiment 2: commuting versus non-commuting pieces.  We apply Lie--Trotter
# splitting with highly accurate sub-integrators to the linear problem
#    y' = (J1 + J2) y,  t in [0,1],  y(0) = [1, 1],
# with
#    J1 = [-1  0]     J2 = [-2  eps]
#         [ 0 -2],         [ 0   -1],
# so that ||J1 J2 - J2 J1|| = |eps|.  For eps = 0 the pieces commute and
# Lie--Trotter splitting is exact (up to the sub-integrator error); otherwise
# its error constant scales with the commutator.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
from scipy.linalg import expm
import sys
sys.path.append('../04_explicit_one_step')
from ERK import *
from FractionalStep import *

# fractional-step methods to test, from the catalogue in FractionalStep.py
methods = [('Lie-Trotter', LieTrotter()),
           ('Lie-Trotter adjoint', LieTrotterAdjoint()),
           ('Strang-Marchuk', StrangMarchuk()),
           ('OS2(2,2)-1/4', OS2(0.25)),
           ('OS2(2,2)-2', OS2(2.0)),
           ('Ruth', Ruth())]

# step sizes to test
Hvals = 0.05 / 2.0**np.arange(7)

# test runner function: returns the errors and the convergence orders estimated
# from each pair of successive step sizes
def RunTest(S, f, B, hsub, tspan, y0, ref):
    """
    Runs the fractional-step method with coefficient table S, using an ERK
    method with Butcher table B[l] and step size hsub for each partition l,
    with right-hand side functions f[l], at each step size in Hvals.
    """
    errs = np.zeros(Hvals.size)
    for idx, H in enumerate(Hvals):
        solvers = [ERK(f[l], B[l], hsub) for l in range(len(f))]
        stepper = FractionalStep(S, solvers)
        Y, success = stepper.Evolve(tspan, y0, H)
        errs[idx] = np.linalg.norm(Y[-1,:] - ref, np.inf)
    orders = np.log(errs[0:-1]/errs[1:])/np.log(Hvals[0:-1]/Hvals[1:])
    return errs, orders


#### Experiment 1 ####
print("Experiment 1: splitting order versus sub-integrator order")

# KPR problem parameters
t0 = 0.0
tf = 5.0
G = -1.0
e = 0.5
w = 2.0
tspan = np.array([t0, tf])

# KPR component functions
def r(t):
    return 0.5*np.cos(t)
def s(t):
    return np.cos(w*t)
def rdot(t):
    return -0.5*np.sin(t)
def sdot(t):
    return -w*np.sin(w*t)

# KPR true solution
def ytrue(t):
    return np.array((np.sqrt(1+r(t)), np.sqrt(2+s(t))))
y0 = ytrue(t0)
ref = ytrue(tf)

# KPR right-hand side pieces: f1 advances only u, and f2 advances only v
def f1(t, y):
    """ First piece (first row) of the right-hand side """
    u = y[0]
    v = y[1]
    return (np.array([[G, e], [0, 0]])
            @ np.array([(-1 + u**2 - r(t)) / (2 * u),
                        (-2 + v**2 - s(t)) / (2 * v)])
            + np.array([rdot(t) / (2 * u), 0]))
def f2(t, y):
    """ Second piece (second row) of the right-hand side """
    u = y[0]
    v = y[1]
    return (np.array([[0, 0], [e, -1]])
            @ np.array([(-1 + u**2 - r(t)) / (2 * u),
                        (-2 + v**2 - s(t)) / (2 * v)])
            + np.array([0, sdot(t) / (2 * v)]))

# sub-integrators: (name, Butcher table, sub-integrator step size); a step
# size of tf-t0 takes a single step per fractional step
subintegrators = [('accurate RK4', ERK4(), 1e-3),
                  ('forward Euler', ERK1(), tf-t0),
                  ('Heun', Heun(), tf-t0),
                  ('RK3', ERK3(), tf-t0)]
for subname, B, hsub in subintegrators:
    print("\n  sub-integrator = %s:" % (subname))
    for name, S in methods:
        errs, orders = RunTest(S, [f1, f2], [B, B], hsub, tspan, y0, ref)
        print("    %-20s (splitting order %i):  min err = %8.2e,  estimated orders:  %s" %
              (name, S['p'], errs[-1], ' '.join('%5.2f' % q for q in orders)))


#### Experiment 2 ####
print("\nExperiment 2: commuting versus non-commuting pieces (Lie-Trotter, accurate RK4)")

# problem time interval and initial condition
t0 = 0.0
tf = 1.0
y0 = np.array([1.0, 1.0])
tspan = np.array([t0, tf])
J1 = np.array([[-1.0, 0.0], [0.0, -2.0]])

for eps in [0.0, 0.25, 0.5, 1.0, 2.0]:
    J2 = np.array([[-2.0, eps], [0.0, -1.0]])
    def f1(t,y):
        """ First piece of the right-hand side """
        return J1 @ y
    def f2(t,y):
        """ Second piece of the right-hand side """
        return J2 @ y
    ref = expm((tf-t0)*(J1+J2)) @ y0
    commutator = np.linalg.norm(J1 @ J2 - J2 @ J1, np.inf)
    errs, orders = RunTest(LieTrotter(), [f1, f2], [ERK4(), ERK4()], 1e-3, tspan, y0, ref)
    print("  eps = %4.2f:  ||[J1,J2]|| = %4.2f,  err/H at H = %.2e:  %8.2e,  estimated orders:  %s" %
          (eps, commutator, Hvals[-1], errs[-1]/Hvals[-1], ' '.join('%5.2f' % q for q in orders)))
