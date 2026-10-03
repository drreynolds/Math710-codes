#!/usr/bin/env python
#
# Script that runs various Lie-Trotter subcycling methods and MRI methods
# on the nonlinear Kvaerno Prothero and Robinson problem:
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
# This script uses Lie-Trotter subcycling methods with components at various
# orders of accuracy to see if/how that affects accuracy.  It also runs MRI of
# various orders of accuracy to see how those compare.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../04_explicit_one_step')
from ERK import *
from LTSubcycling import *
from SMSubcycling import *
from MRI import *

# KPR problem parameters
Tf = 5
Nt = 25
tvals = np.linspace(0, Tf, Nt+1)
e = 0.5
w = 100
G = -10

# slow step sizes to try
Hvals = np.array([0.1, 0.05, 0.025, 0.01, 0.005, 0.0025])

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

def runFamily(name, Hvals, w, Y0, Ytrue, tvals, buildStepper):
    # store errors for convergence-rate estimates
    errs = np.zeros(Hvals.size)
    print("\n%s:" % (name))
    for idx, H in enumerate(Hvals):
        h = H/w
        stepper = buildStepper(h, H)
        fast = stepper.FastSolver
        print("  H = %f, h = %f:" % (H, h))
        Y, success = stepper.Evolve(tvals, Y0)
        errs[idx] = np.linalg.norm(np.abs(Y-Ytrue),np.inf)
        if (not success):
            print("  solve failed")
        print("   steps (s,f) = (%i, %i)  nrhs (s,f) = (%i, %i)  err = %.1e" %
            (stepper.get_num_steps(), fast.get_num_steps(), stepper.get_num_rhs(), fast.get_num_rhs(), errs[idx]))

    if (Hvals.size > 2):
        orders = np.log(errs[:-1]/errs[1:])/np.log(Hvals[:-1]/Hvals[1:])
        print('estimated order: ', np.mean(orders))

# Lie-Trotter subcycling evolves slow dynamics once per macro step and
# fast dynamics with h = H/w substeps.
runFamily('Lie-Trotter-Subcycling-1', Hvals, w, Y0, Ytrue, tvals,
    lambda h, H: LTSubcycling(fs, ERK1(), ERK(ff, ERK1(), h), H))

runFamily('Lie-Trotter-Subcycling-2', Hvals, w, Y0, Ytrue, tvals,
    lambda h, H: LTSubcycling(fs, ERK2(), ERK(ff, ERK2(), h), H))

# Strang-Marchuk variants symmetrize the slow/fast splitting.
runFamily('Strang-Marchuk-Subcycling-1', Hvals, w, Y0, Ytrue, tvals,
    lambda h, H: SMSubcycling(fs, ERK1(), ERK(ff, ERK1(), h), H))

runFamily('Strang-Marchuk-2', Hvals, w, Y0, Ytrue, tvals,
    lambda h, H: SMSubcycling(fs, ERK2(), ERK(ff, ERK2(), h), H))

runFamily('Strang-Marchuk-3', Hvals, w, Y0, Ytrue, tvals,
    lambda h, H: SMSubcycling(fs, ERK3(), ERK(ff, ERK3(), h), H))

# MRI-GARK methods couple slow stages to a fast IVP solve over each stage interval.
runFamily('MRI-GARK-ERK22a', Hvals, w, Y0, Ytrue, tvals,
    lambda h, H: MRI(Y0, fs, ff, MRIGARKERK22a(), ERK(ff, ERK2(), h), H))

runFamily('MRI-GARK-ERK33a', Hvals, w, Y0, Ytrue, tvals,
    lambda h, H: MRI(Y0, fs, ff, MRIGARKERK33a(), ERK(ff, ERK3(), h), H))

runFamily('MRI-GARK-ERK45a', Hvals, w, Y0, Ytrue, tvals,
    lambda h, H: MRI(Y0, fs, ff, MRIGARKERK45a(), ERK(ff, ERK4(), h), H))
