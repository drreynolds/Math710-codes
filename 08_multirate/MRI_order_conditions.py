#!/usr/bin/env python
# Functions to check the order conditions of explicit MRI-GARK methods, to any
# order, through their GARK representation.  Following the lecture notes, we
# replace each exact fast solve by one step of a Runge--Kutta method of
# sufficiently high order; the resulting two-way GARK method (partition 1 =
# slow, partition 2 = fast) is then checked with GARK_order from
# 07_implicit_explicit/GARK_order_conditions.py, which enumerates bicolored
# rooted trees.  A simple "main" checks the methods in MRI.py.
#
# The GARK blocks (see "The GARK Tableau of an MRI-GARK Method" in the notes),
# for an s-stage MRI-GARK method with coupling coefficients gamma_{ij}^{k},
# slow base method (A, b, c), and fast method (AF, bF, cF) with sF stages, are:
#   A^{SS} = A,  b^S = b;
#   b^F = [dc_2 bF; ... ; dc_s bF];
#   A^{FF}: block (i,lam) = dc_i AF if lam = i, dc_lam 1 bF^T if lam < i;
#   A^{SF}: row i, block lam = dc_lam bF^T for 2 <= lam <= i;
#   A^{FS}: block row i, column j = a_{i-1,j} 1 + sum_k gamma_{ij}^{k} AF CF^k 1,
# where dc_i = c_i - c_{i-1}, CF = diag(cF), and the fast stages are grouped
# by slow stage i = 2,...,s.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

# general imports
import math
import numpy as np
import sympy as sp
import os
import sys
sys.path.append('../07_implicit_explicit')
from GARK_order_conditions import GARK_order

def Gauss_tableau(sF):
    ''' Usage: B = Gauss_tableau(sF)

        Returns the Butcher table of the sF-stage Gauss--Legendre method, of
        order 2*sF, in double precision, as a dictionary with fields 'A',
        'b' and 'c'.  The nodes c are the Gauss points on [0,1], and as a
        collocation method its coefficients are
          b_j = int_0^1 l_j(tau) dtau,  a_ij = int_0^{c_i} l_j(tau) dtau,
        where l_j is the Lagrange basis polynomial for the nodes c.  The b_j
        are the Gauss weights themselves, and since each l_j has degree
        sF-1, a_ij is computed exactly by the same Gauss rule mapped to
        [0,c_i].  (Solving the equivalent collocation conditions with the
        Vandermonde matrix of the nodes would lose accuracy as sF grows,
        since that matrix is ill-conditioned.)
    '''
    x, w = np.polynomial.legendre.leggauss(sF)
    c = (x + 1)/2
    b = w/2

    # utility routine to evaluate l_j at the points tau
    def lagrange(j, tau):
        lj = np.ones_like(tau)
        for m in range(sF):
            if (m != j):
                lj *= (tau - c[m])/(c[j] - c[m])
        return lj

    A = np.zeros((sF, sF))
    for i in range(sF):
        tau = c[i]*c                     # Gauss points mapped to [0,c_i]
        for j in range(sF):
            A[i,j] = c[i]*np.dot(b, lagrange(j, tau))
    return {'A': A, 'b': b, 'c': c}

def MRI_base_method(C):
    ''' Usage: A, b, c = MRI_base_method(C)

        Returns the slow base method of the MRI-GARK method with coupling
        table C (as in MRI.py), using a_{ij} = sum_{lam<=i} gbar_{lam,j}, with
        gbar_{ij} = sum_k gamma_{ij}^{k}/(k+1), and b^T = e_s^T A.
    '''
    G = np.asarray(C['G'], dtype=float)
    c = np.asarray(C['c'], dtype=float)
    K = np.size(G, 0)
    Gbar = sum(G[k]/(k+1) for k in range(K))
    A = np.cumsum(Gbar, axis=0)
    b = A[-1,:].copy()
    return A, b, c

def MRI_tableau(C, BF):
    ''' Usage: A, b = MRI_tableau(C, BF)

        Inputs:
          C is an explicit MRI-GARK coupling table (as in MRI.py), with
             fields 'G' (gamma_{ij}^{k} stored as G[k,i,j]) and 'c'
          BF is the Butcher table of the fast method, with fields 'A', 'b'
             and 'c'

        Outputs:
          A, b are the GARK coefficient blocks and solution weights of the
             resulting two-way GARK method (partition 1 = slow, partition
             2 = fast), in the format used by GARK_order
    '''
    G = np.asarray(C['G'], dtype=float)
    K = np.size(G, 0)
    AS, bS, c = MRI_base_method(C)
    s = c.size
    dc = np.diff(c)                  # dc[i-1] = c_i - c_{i-1}, i = 1,...,s-1 (0-based i)
    AF = np.asarray(BF['A'], dtype=float)
    bF = np.asarray(BF['b'], dtype=float)
    cF = np.asarray(BF['c'], dtype=float)
    sF = bF.size
    nF = (s-1)*sF                    # total number of fast stages
    ones = np.ones(sF)

    # fast stages within slow stage i (0-based i = 1,...,s-1) are rows rows(i)
    def rows(i):
        return slice((i-1)*sF, i*sF)

    # fast-fast block, and fast weights
    AFF = np.zeros((nF, nF))
    for i in range(1, s):
        AFF[rows(i), rows(i)] = dc[i-1]*AF
        for lam in range(1, i):
            AFF[rows(i), rows(lam)] = dc[lam-1]*np.outer(ones, bF)
    bFF = np.concatenate([dc[i-1]*bF for i in range(1, s)])

    # slow-fast block
    ASF = np.zeros((s, nF))
    for i in range(1, s):
        for lam in range(1, i+1):
            ASF[i, rows(lam)] = dc[lam-1]*bF

    # fast-slow block
    AFS = np.zeros((nF, s))
    for i in range(1, s):
        for j in range(s):
            col = AS[i-1, j]*ones
            for k in range(K):
                col = col + G[k,i,j]*(AF @ np.power(cF, k))
            AFS[rows(i), j] = col

    return [[AS, ASF], [AFS, AFF]], [bS, bFF]

def MRI_order(C, maxorder=4, tol=1e-10, verbose=False):
    ''' Usage: pbase, p, failed = MRI_order(C, maxorder, tol, verbose)

        Returns the order pbase of the slow base method, and the order p of
        the MRI-GARK method with coupling table C, together with the labels
        of the lowest-order failing GARK conditions.  Only conditions through
        order maxorder are checked, so if every one of them holds then the
        method has order AT LEAST maxorder; in that case p = maxorder, and a
        warning is printed, since a higher maxorder is needed to determine
        the order exactly.

        The fast method is the Gauss--Legendre method with enough stages
        that its own order conditions hold for every fast tree that appears
        in the GARK conditions through order maxorder: in an order-q tree,
        each of the at most q-1 fast-slow edges replaces a single weight by
        AF CF^k 1, adding at most K-1 vertices to the corresponding fast
        tree (where K is the number of Gamma matrices), so a fast method of
        order maxorder + (maxorder-1)*(K-1) suffices.
    '''
    K = np.size(np.asarray(C['G']), 0)
    pF = maxorder + (maxorder-1)*(K-1)
    BF = Gauss_tableau(math.ceil(pF/2))
    A, b = MRI_tableau(C, BF)
    pbase, _ = GARK_order([[sp.Matrix(A[0][0])]], [sp.Matrix(b[0])], tol=tol, maxorder=maxorder)
    p, failed = GARK_order(A, b, tol=tol, maxorder=maxorder, verbose=verbose)
    if (p == maxorder):
        print("  MRI_order warning: all conditions through order %i hold, so the order is at least %i;" % (maxorder, maxorder))
        print("    call with a larger maxorder to determine it exactly")
    return pbase, p, failed


# main routine
if __name__ == '__main__':
    from MRI import MRIGARKERK22a, MRIGARKERK33a, MRIGARKERK45a

    maxorder = 5
    tests = [('MRI-GARK-ERK22a', MRIGARKERK22a()),
             ('MRI-GARK-ERK33a', MRIGARKERK33a()),
             ('MRI-GARK-ERK45a', MRIGARKERK45a())]
    for name, C in tests:
        print("\n", name, ":", sep='')
        pbase, p, failed = MRI_order(C, maxorder=maxorder, tol=1e-8)
        print("  slow base method order = %i,  MRI-GARK order = %i" % (pbase, p))
        if (p < min(pbase, maxorder)):
            print("  failing coupling conditions: %s" % (', '.join(failed)))
