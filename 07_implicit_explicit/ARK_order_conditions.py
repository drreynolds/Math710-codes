#!/usr/bin/env python
# Functions to check the order conditions (including the coupling conditions)
# and the stiff-limit amplification function for two-component additive
# Runge--Kutta (ARK) methods.  Includes a simple "main" that uses these
# functions to examine a few ARK methods from the lecture notes.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

# general imports
import numpy as np
import sympy as sp
from itertools import product

def ARK_order_conditions(BE, BI, maxorder=4):
    ''' Usage: res = ARK_order_conditions(BE, BI, maxorder)

        Inputs:
          BE is the explicit Butcher table, with components:
             BE['A'] -- the Butcher table matrix
             BE['b'] -- the solution coefficients
          BI is the implicit Butcher table, with the same components
          maxorder is optional, specifying the highest order of
             conditions to generate (1 <= maxorder <= 5, default 4)

        Outputs:
          res is a dictionary indexed by order q = 1,...,maxorder, where
             res[q] is itself a dictionary mapping a label for each order-q
             condition to its residual (zero when the condition holds)

        The abscissae for each table are computed as c = A*1 (internal
        consistency), so that the tables need not share their abscissae.
        Condition labels list the table used for each factor, e.g.,
        'bE.AI.cE' denotes (bE)^T AI cE - 1/6, and 'bI.cE*cI' denotes
        (bI)^T diag(cE) cI - 1/3; each '*' applies to everything on its
        right, so parentheses mark the one order-5 product of two matrix-
        vector terms, e.g., 'bE.(AE.cI)*(AI.cE)' denotes
        (bE)^T diag(AE cI) AI cE - 1/20.  Labels where all factors use the same
        table are the order conditions for that table alone; all others
        are coupling conditions.  Inputs may be numpy arrays of floats,
        or sympy matrices with exact (rational or symbolic) entries.
    '''

    # set up tables and abscissae
    A = {'E': sp.Matrix(BE['A']), 'I': sp.Matrix(BI['A'])}
    b = {'E': sp.Matrix(BE['b']), 'I': sp.Matrix(BI['b'])}
    s = A['E'].shape[0]
    if ((A['I'].shape[0] != s) or (b['E'].shape[0] != s) or (b['I'].shape[0] != s)):
        raise ValueError("ARK_order_conditions ERROR: incompatible Butcher tables supplied")
    one = sp.ones(s,1)
    c = {k: A[k]*one for k in 'EI'}
    R = sp.Rational
    def dot(u,v):
        return sum(u[i]*v[i] for i in range(s))
    def hadamard(u,v):
        return sp.Matrix([u[i]*v[i] for i in range(s)])

    # generate conditions
    res = {q: {} for q in range(1,maxorder+1)}
    for u in 'EI':
        res[1][f'b{u}.1'] = dot(b[u],one) - 1
    if (maxorder >= 2):
        for u, x in product('EI', repeat=2):
            res[2][f'b{u}.c{x}'] = dot(b[u],c[x]) - R(1,2)
    if (maxorder >= 3):
        for u, x, y in product('EI', repeat=3):
            if (x <= y):
                res[3][f'b{u}.c{x}*c{y}'] = dot(b[u],hadamard(c[x],c[y])) - R(1,3)
            res[3][f'b{u}.A{x}.c{y}'] = dot(b[u],A[x]*c[y]) - R(1,6)
    if (maxorder >= 4):
        for u, x, y, z in product('EI', repeat=4):
            if (x <= y <= z):
                res[4][f'b{u}.c{x}*c{y}*c{z}'] = dot(b[u],hadamard(c[x],hadamard(c[y],c[z]))) - R(1,4)
            res[4][f'b{u}.c{x}*A{y}.c{z}'] = dot(b[u],hadamard(c[x],A[y]*c[z])) - R(1,8)
            if (y <= z):
                res[4][f'b{u}.A{x}.c{y}*c{z}'] = dot(b[u],A[x]*hadamard(c[y],c[z])) - R(1,12)
            res[4][f'b{u}.A{x}.A{y}.c{z}'] = dot(b[u],A[x]*A[y]*c[z]) - R(1,24)
    if (maxorder >= 5):
        for u, x, y, z, w in product('EI', repeat=5):
            if (x <= y <= z <= w):
                res[5][f'b{u}.c{x}*c{y}*c{z}*c{w}'] = dot(b[u],hadamard(c[x],hadamard(c[y],hadamard(c[z],c[w])))) - R(1,5)
            if (x <= y):
                res[5][f'b{u}.c{x}*c{y}*A{z}.c{w}'] = dot(b[u],hadamard(c[x],hadamard(c[y],A[z]*c[w]))) - R(1,10)
            if (z <= w):
                res[5][f'b{u}.c{x}*A{y}.c{z}*c{w}'] = dot(b[u],hadamard(c[x],A[y]*hadamard(c[z],c[w]))) - R(1,15)
            res[5][f'b{u}.c{x}*A{y}.A{z}.c{w}'] = dot(b[u],hadamard(c[x],A[y]*A[z]*c[w])) - R(1,30)
            if ((x,y) <= (z,w)):
                res[5][f'b{u}.(A{x}.c{y})*(A{z}.c{w})'] = dot(b[u],hadamard(A[x]*c[y],A[z]*c[w])) - R(1,20)
            if (y <= z <= w):
                res[5][f'b{u}.A{x}.c{y}*c{z}*c{w}'] = dot(b[u],A[x]*hadamard(c[y],hadamard(c[z],c[w]))) - R(1,20)
            res[5][f'b{u}.A{x}.c{y}*A{z}.c{w}'] = dot(b[u],A[x]*hadamard(c[y],A[z]*c[w])) - R(1,40)
            if (z <= w):
                res[5][f'b{u}.A{x}.A{y}.c{z}*c{w}'] = dot(b[u],A[x]*A[y]*hadamard(c[z],c[w])) - R(1,60)
            res[5][f'b{u}.A{x}.A{y}.A{z}.c{w}'] = dot(b[u],A[x]*A[y]*A[z]*c[w]) - R(1,120)

    # simplify residuals
    for q in res:
        for key in res[q]:
            res[q][key] = sp.simplify(res[q][key])
    return res


def ARK_order(BE, BI, tol=1e-10, maxorder=4, verbose=False):
    ''' Usage: p, failed = ARK_order(BE, BI, tol, maxorder, verbose)

        Inputs:
          BE, BI are the explicit and implicit Butcher tables (see
             ARK_order_conditions)
          tol is optional, specifying the tolerance for considering a
             residual to be zero (default 1e-10)
          maxorder is optional, specifying the highest order to check
             (default 4)
          verbose is optional, printing the failing conditions if True

        Outputs:
          p is the order of the ARK method (or maxorder, if all checked
             conditions hold)
          failed is a list of the labels of the lowest-order failing
             conditions (empty if p == maxorder)
    '''
    res = ARK_order_conditions(BE, BI, maxorder)
    p = 0
    for q in range(1,maxorder+1):
        failed = [key for key, val in res[q].items() if abs(complex(sp.N(val))) > tol]
        if (len(failed) > 0):
            if (verbose):
                print("  failing order-%i conditions:" % (q), failed)
            return p, failed
        p = q
    return p, []


def ARK_stiff_limit(BE, BI):
    ''' Usage: R, Rlim = ARK_stiff_limit(BE, BI)

        Inputs:
          BE, BI are the explicit and implicit Butcher tables (see
             ARK_order_conditions)

        Outputs:
          R is the ARK amplification function R(zE,zI) as a sympy expression
             in the symbols zE and zI, where
                R(zE,zI) = 1 + (zE*bE + zI*bI)^T (I - zE*AE - zI*AI)^{-1} 1
          Rlim is the limit of R(zE,zI) as zI -> -infinity, as a function of zE

        For the combined method to be "L-stable" in the sense of Kennedy &
        Carpenter, Rlim should equal zero for every zE.
    '''
    zE, zI = sp.symbols('zE zI')
    AE = sp.Matrix(BE['A']); bE = sp.Matrix(BE['b'])
    AI = sp.Matrix(BI['A']); bI = sp.Matrix(BI['b'])
    s = AE.shape[0]
    M = sp.eye(s) - zE*AE - zI*AI
    R = sp.simplify(1 + ((zE*bE + zI*bI).T * M.LUsolve(sp.ones(s,1)))[0])
    Rlim = sp.simplify(sp.limit(R, zI, -sp.oo))
    return R, Rlim


# main routine
if __name__ == '__main__':

    R = sp.Rational

    # ARS (1,2,2): explicit and (padded) implicit midpoint
    BE = {'A': sp.Matrix([[0, 0], [R(1,2), 0]]), 'b': sp.Matrix([0, 1])}
    BI = {'A': sp.Matrix([[0, 0], [0, R(1,2)]]), 'b': sp.Matrix([0, 1])}
    ARS122 = (BE, BI)

    # ARS (2,2,2)
    g = (2 - sp.sqrt(2))/2
    d = 1 - 1/(2*g)
    BE = {'A': sp.Matrix([[0, 0, 0], [g, 0, 0], [d, 1-d, 0]]), 'b': sp.Matrix([d, 1-d, 0])}
    BI = {'A': sp.Matrix([[0, 0, 0], [0, g, 0], [0, 1-g, g]]), 'b': sp.Matrix([0, 1-g, g])}
    ARS222 = (BE, BI)

    # ARS (3,4,3), using the 10-digit coefficients from Ascher et al. (1997)
    g = 0.4358665215
    b1 = 1.208496649
    b2 = -0.644363171
    BE = {'A': np.array([[0.0, 0.0, 0.0, 0.0],
                         [g, 0.0, 0.0, 0.0],
                         [0.3212788860, 0.3966543747, 0.0, 0.0],
                         [-0.105858296, 0.5529291479, 0.5529291479, 0.0]]),
          'b': np.array([0.0, b1, b2, g])}
    BI = {'A': np.array([[0.0, 0.0, 0.0, 0.0],
                         [0.0, g, 0.0, 0.0],
                         [0.0, 0.2820667392, g, 0.0],
                         [0.0, b1, b2, g]]),
          'b': np.array([0.0, b1, b2, g])}
    ARS343 = (BE, BI)

    # coupling-failure demonstration pairs from the lecture notes
    Heun = {'A': sp.Matrix([[0, 0], [1, 0]]), 'b': sp.Matrix([R(1,2), R(1,2)])}
    RK4 = {'A': sp.Matrix([[0, 0, 0, 0], [R(1,2), 0, 0, 0], [0, R(1,2), 0, 0], [0, 0, 1, 0]]),
           'b': sp.Matrix([R(1,6), R(1,3), R(1,3), R(1,6)])}
    ERK3 = {'A': sp.Matrix([[0, 0, 0, 0], [R(1,2), 0, 0, 0], [0, R(1,2), 0, 0], [1, 0, 0, 0]]),
            'b': sp.Matrix([R(1,6), 0, R(2,3), R(1,6)])}
    ESDIRK3 = {'A': sp.Matrix([[0, 0, 0, 0], [R(1,6), R(1,3), 0, 0],
                               [R(1,2), -R(1,3), R(1,3), 0], [-R(2,3), R(2,3), R(2,3), R(1,3)]]),
               'b': sp.Matrix([R(1,6), 0, R(2,3), R(1,6)])}

    tests = [('ARS(1,2,2)', ARS122), ('ARS(2,2,2)', ARS222), ('ARS(3,4,3)', ARS343),
             ('Heun + implicit midpoint', (Heun, ARS122[1])),
             ('RK4 + ESDIRK3', (RK4, ESDIRK3)), ('ERK3 + ESDIRK3', (ERK3, ESDIRK3))]
    for name, (BE, BI) in tests:
        print("\n", name, ":", sep='')
        pE, _ = ARK_order(BE, BE, tol=1e-8)
        pI, _ = ARK_order(BI, BI, tol=1e-8)
        p, failed = ARK_order(BE, BI, tol=1e-8)
        print("  explicit order = %i,  implicit order = %i,  ARK order = %i" % (pE, pI, p))
        if (p < min(pE, pI)):
            print("  failing coupling conditions:", failed)
        Rf, Rlim = ARK_stiff_limit(BE, BI)
        print("  R(zE, zI -> -infinity) =", sp.N(sp.expand(Rlim), 6))
