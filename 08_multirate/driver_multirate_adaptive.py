#!/usr/bin/env python
#
# Script that runs subcycling and MRI methods with an adaptive fast solver on
# the nonlinear Kvaerno Prothero and Robinson problem:
#    [u]' = [ G  e ] [(-1+u^2-r)/(2u)] + [ r'(t)/(2u) ]
#    [v]    [ e -1 ] [(-2+v^2-s)/(2v)]   [ s'(t)/(2v) ]
#         = [fs(t,y)]
#           [ff(t,y)]
# where r(t) = 0.5*cos(t),  s(t) = cos(w*t),  0 < t < 5.
# This problem has analytical solution given by
#    u(t) = sqrt(1+r(t)),  v(t) = sqrt(2+s(t)).
#
# We use the parameters:
#   e = inter-variable coupling strength (0.5)
#   G = stiffness at slow time scale (-10)
#   w = variable time-scale separation factor (100)
#
# For each of a few slow step sizes H, this script runs Lie-Trotter and
# Strang-Marchuk subcycling (built from 07_implicit_explicit/FractionalStep.py,
# with one explicit slow step per sub-step) and the MRI-GARK-ERK33a method,
# all using the same adaptive fast solver, over a range of fast relative
# tolerances.  As the fast tolerance is tightened, the errors of the
# subcycling methods level off at their splitting error, no matter how
# accurately the fast piece is solved, while the error of the MRI method
# keeps decreasing until it reaches its own (much smaller) O(H^3) error, at a
# similar cost.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import sys

sys.path.append('../04_explicit_one_step')
from ERK import *
from AdaptERK import *
sys.path.append('../07_implicit_explicit')
from FractionalStep import *
from MRI import *

# KPR problem parameters
Tf = 5
Nt = 25
tvals = np.linspace(0, Tf, Nt+1)
e = 0.5
w = 100
G = -10

# slow method for the subcycling methods, and adaptive fast method
slow_table = ERK2()
fast_adapt_table = BogackiShampine()

# slow step sizes and fast relative tolerances to try
Hvals = np.array([0.02, 0.01])
rtols = 10.0**(-np.arange(3, 11))
atol = 1e-12

# KPR component functions
def r(t):
    return 0.5*np.cos(t)
def s(t):
    return np.cos(w*t)
def rdot(t):
    return -0.5*np.sin(t)
def sdot(t):
    return -w*np.sin(w*t)

# KPR true solution functions
def utrue(t):
    return np.sqrt(1+r(t))
def vtrue(t):
    return np.sqrt(2+s(t))
def ytrue(t):
    return np.array((utrue(t), vtrue(t)))

# initial condition
Y0 = ytrue(0)

# true solution at each output time
Ytrue = np.zeros((Nt+1,2))
for i in range(Nt+1):
    Ytrue[i,:] = ytrue(tvals[i])

# KPR right-hand side functions
def fs(t, y):
    u = y[0]
    v = y[1]
    return (np.array([[G, e], [0, 0]])
            @ np.array([(-1 + u**2 - r(t)) / (2 * u),
                        (-2 + v**2 - s(t)) / (2 * v)])
            + np.array([rdot(t) / (2 * u), 0]))
def ff(t, y):
    u = y[0]
    v = y[1]
    return (np.array([[0, 0], [e, -1]])
            @ np.array([(-1 + u**2 - r(t)) / (2 * u),
                        (-2 + v**2 - s(t)) / (2 * v)])
            + np.array([0, sdot(t) / (2 * v)]))

# utility routines to construct each method from the slow and fast right-hand
# side functions, initial condition, slow and adaptive fast Butcher tables,
# slow step size H, and fast tolerances rtol and atol; each returns the
# stepper, the slow solver, and the fast solver (so that their right-hand
# side evaluations may be counted separately)
def LTSubcycling(fs, ff, Y0, Bs, Bf, H, rtol, atol):
    # Lie-Trotter subcycling: one slow step over [t_n, t_n+H], followed by a
    # fast solve over the same interval
    slow = ERK(fs, Bs, H)
    fast = AdaptERK(ff, Y0, Bf, rtol=rtol, atol=atol)
    return FractionalStep(LieTrotter(), [slow, fast], H), slow, fast
def SMSubcycling(fs, ff, Y0, Bs, Bf, H, rtol, atol):
    # Strang-Marchuk subcycling: fast solves over each half of [t_n, t_n+H],
    # surrounding one slow step over the full interval (so the fast piece is
    # partition 1, which takes the two half steps)
    slow = ERK(fs, Bs, H)
    fast = AdaptERK(ff, Y0, Bf, rtol=rtol, atol=atol)
    return FractionalStep(StrangMarchuk(), [fast, slow], H), slow, fast
def MRIMethod(fs, ff, Y0, Bs, Bf, H, rtol, atol):
    # (Bs is unused, since the MRI method has its own slow coupling table)
    fast = AdaptERK(ff, Y0, Bf, rtol=rtol, atol=atol)
    stepper = MRI(Y0, fs, ff, MRIGARKERK33a(), fast, H)
    return stepper, stepper, fast

methods = [('Lie-Trotter subcycling', LTSubcycling),
           ('Strang-Marchuk subcycling', SMSubcycling),
           ('MRI-GARK-ERK33a', MRIMethod)]

for H in Hvals:
    print("\nH = %g:" % (H))
    for name, buildMethod in methods:
        print("  %s:" % (name))
        for rtol in rtols:
            stepper, slow, fast = buildMethod(fs, ff, Y0, slow_table, fast_adapt_table, H, rtol, atol)
            Y, success = stepper.Evolve(tvals, Y0)
            err = np.linalg.norm(np.abs(Y-Ytrue),np.inf)
            if (not success):
                print("    rtol = %.0e:  solve failed" % (rtol))
                continue
            print("    rtol = %.0e:  nrhs (s,f) = (%5i, %7i)  err = %.2e" %
                  (rtol, slow.get_num_rhs(), fast.get_num_rhs(), err))
