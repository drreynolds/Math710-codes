#!/usr/bin/env python
# Functions to plot the linear stability regions of fractional-step
# Runge--Kutta (FSRK) methods, i.e., fractional-step (operator-splitting)
# methods in which each sub-flow is approximated by a Runge--Kutta method.
# For the scalar test problem
#      y' = lambda^{1} y + ... + lambda^{M} y,
# each sub-step simply multiplies the solution by the stability function of
# its sub-integrator, so that the FSRK stability function is the product
#      R = prod_k prod_l R^{l}( alpha_k^{l} z^{l} ),   z^{l} = H lambda^{l}
# (Spiteri & Wei, J. Comput. Phys. 476:111900, 2023, Theorem 3.2).  To plot
# this in a single complex variable z, we write z^{l} = mu_l z for given
# multipliers mu_l, as in Spiteri & Wei's examples.
#
# A simple "main" reproduces two of their examples (experiments 3 and 4 in
# the lecture notes):
#   1. Strang--Marchuk splitting of y' = -20 y with Heun's method for
#      partition 1 and the L-stable SDIRK(2,2) method for partition 2, for
#      the 50-50, 10-90 and 90-10 splits (their Figure 1), and with the roles
#      of the two sub-integrators exchanged;
#   2. Ruth's splitting with Kutta's RK3 method for partition 1 and the
#      SDIRK(2,3) method for partition 2, with lambda^{1} = lambda^{2} (their
#      Figure 5), and with the sub-integrators exchanged.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

# general imports
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../04_explicit_one_step')
sys.path.append('../05_implicit_one_step')
sys.path.append('../utilities')
from ERK import Heun, ERK3
from DIRK import SDIRK2
from RK_stability import RK_stability_function
from FractionalStep import StrangMarchuk, Ruth

def FSRK_stability_function(S, B, mu, z):
    """
    Usage: R = FSRK_stability_function(S, B, mu, z)

    Evaluates the stability function of the FSRK method with fractional-step
    coefficient table S (from FractionalStep.py), where partition l uses
    the Runge--Kutta method with Butcher table B[l], at every entry of the
    array z, where z^{l} = mu[l] z.
    """
    alpha = np.asarray(S['alpha'], dtype=float)
    R = np.ones(np.shape(z), dtype=complex)
    for k in range(np.size(alpha,1)):
        for l in range(np.size(alpha,0)):
            if (alpha[l,k] != 0.0):
                R *= RK_stability_function(B[l], alpha[l,k]*mu[l]*z)
    return R

def FSRK_poles(S, B, mu):
    """
    Usage: poles = FSRK_poles(S, B, mu)

    Returns the poles of the FSRK stability function in the variable z.  The
    stability function of each Runge--Kutta method has poles at z = 1/a,
    for each nonzero eigenvalue a of its matrix A, and so the sub-step
    with coefficient alpha_k^{l} contributes poles at z = 1/(alpha_k^{l} mu_l a).
    These lie in the left half-plane when alpha_k^{l} < 0 (for a > 0).
    """
    alpha = np.asarray(S['alpha'], dtype=float)
    poles = []
    for l in range(np.size(alpha,0)):
        evals = np.linalg.eigvals(np.asarray(B[l]['A'], dtype=float))
        for a in evals[np.abs(evals) > 1e-14]:
            for k in range(np.size(alpha,1)):
                if (alpha[l,k] != 0.0):
                    poles.append(1/(alpha[l,k]*mu[l]*a))
    return np.unique(np.round(np.array(poles), 12))

def FSRK_stab_region(S, B, mu, box, ax, color, label):
    """
    Usage: FSRK_stab_region(S, B, mu, box, ax, color, label)

    Plots the boundary of the stability region { z : |R(z)| <= 1 } of the FSRK
    method (S, B) with multipliers mu, over the box [xl, xr, yl, yr] in the
    complex plane, on the matplotlib axes ax, using the given color and
    legend label.  Poles in the left half-plane are marked with an 'x',
    since a stability region may have holes too small to resolve on the mesh.
    """
    N = 801
    x = np.linspace(box[0], box[1], N)
    y = np.linspace(box[2], box[3], N)
    X, Y = np.meshgrid(x, y)
    R = np.abs(FSRK_stability_function(S, B, mu, X + Y*1j))
    ax.contour(x, y, R, levels=[1.0], colors=[color], linewidths=2)
    ax.plot([], [], color=color, linewidth=2, label=label)
    for p in FSRK_poles(S, B, mu):
        if ((p.real < 0) and (box[0] <= p.real <= box[1]) and (box[2] <= p.imag <= box[3])):
            ax.plot(p.real, p.imag, 'x', color=color, markersize=8)


# main routine
if __name__ == '__main__':

    # sub-integrators
    gamma22 = (2 - np.sqrt(2))/2       # L-stable SDIRK(2,2)
    gamma23 = (3 + np.sqrt(3))/6       # third-order SDIRK(2,3)
    SDIRK22 = SDIRK2(gamma22)
    SDIRK23 = SDIRK2(gamma23)
    HeunB = Heun()
    RK3 = ERK3()

    # Experiment 3: Strang--Marchuk splitting of y' = -20 y, with
    # z^{1} = mu_1 z for the Heun partition and z^{2} = mu_2 z for the SDIRK
    # partition, where z = -H as in Spiteri & Wei
    splits = [('50-50 split', (10.0, 10.0)),
              ('10-90 split', (2.0, 18.0)),
              ('90-10 split', (18.0, 2.0))]
    colors = ['tab:red', 'tab:green', 'tab:blue']
    box = [-3.8, 0.6, -4.5, 4.5]
    box2 = [-3.8, 6.5, -4.5, 4.5]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for (name, mu), color in zip(splits, colors):
        # Heun takes the two half steps, SDIRK(2,2) takes the full step
        FSRK_stab_region(StrangMarchuk(), [HeunB, SDIRK22], mu, box, axs[0], color, name)
        # roles exchanged: SDIRK(2,2) takes the two half steps, Heun the full step
        FSRK_stab_region(StrangMarchuk(), [SDIRK22, HeunB], (mu[1], mu[0]), box2, axs[1], color, name)
    # with the roles exchanged, the 10-90 stability region is unbounded: it is
    # the EXTERIOR of its curve, since R tends to 2(1-2 gamma)^2/(81 gamma^4) < 1
    # as |z| -> infinity
    print("Strang, SDIRK(2,2) half steps + Heun, 10-90 split: the stability region is the exterior of its curve")
    axs[0].set_title('Strang: Heun (half steps) + SDIRK(2,2)')
    axs[1].set_title('Strang: SDIRK(2,2) (half steps) + Heun')
    for ax in axs:
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':')
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('FSRK_Strang_stability.png')

    # Experiment 4: Ruth splitting with lambda^{1} = lambda^{2}, so z^{1} = z^{2} = z
    box = [-10, 5, -10, 10]
    fig, axs = plt.subplots(1, 2, figsize=(11, 6.5))
    FSRK_stab_region(Ruth(), [RK3, SDIRK23], (1.0, 1.0), box, axs[0], 'tab:blue', 'Ruth (RK3 + SDIRK(2,3))')
    FSRK_stab_region(Ruth(), [SDIRK23, RK3], (1.0, 1.0), box, axs[1], 'tab:blue', 'Ruth (SDIRK(2,3) + RK3)')
    for ax in axs:
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':')
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('FSRK_Ruth_stability.png')

    # report the left half-plane poles for experiment 4
    for name, B in [('RK3 + SDIRK(2,3)', [RK3, SDIRK23]), ('SDIRK(2,3) + RK3', [SDIRK23, RK3])]:
        poles = FSRK_poles(Ruth(), B, (1.0, 1.0))
        poles = poles[poles.real < 0]
        print("Ruth (%s): poles in the left half-plane at z = %s" %
              (name, ', '.join('%.4g' % p.real if abs(p.imag) < 1e-12 else '%.4g%+.4gi' % (p.real, p.imag) for p in poles)))

    plt.show()
