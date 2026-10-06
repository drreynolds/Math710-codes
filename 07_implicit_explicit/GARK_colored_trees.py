#!/usr/bin/env python
# Functions to generate the order conditions for generalized-structure
# additive Runge--Kutta (GARK) methods with an arbitrary number M of
# partitions, of any order, by enumerating rooted trees.  Each condition is
# written once per rooted tree, using placeholder colors (sigma, nu, mu, ...)
# for its vertices, and must hold for every choice of these colors in
# {1,...,M}.  Optionally applies internal consistency, c^{sigma,nu} =
# c^{sigma}, under which the leaves no longer need colors of their own.
# Includes a simple "main" that uses these functions to print the conditions,
# and to count the distinct and coupling conditions at each order, both as
# polynomials in M and for specific values of M.  The rooted trees themselves
# are enumerated by rooted_trees, in utilities/rooted_trees.py.
#
# Each vertex of a tree carries a color.  The root, of color sigma,
# contributes the solution weights b^{sigma}; a vertex of color nu attached to
# a parent of color mu contributes A^{mu,nu}; and so a leaf of color nu
# attached to a parent of color mu contributes c^{mu,nu} = A^{mu,nu}*1.
#
# In condition labels, '.' denotes a matrix product, e.g., 'b{σ}.A{σ,ν}.c{ν,μ}'
# denotes (b^{σ})^T A^{σ,ν} c^{ν,μ}.  A componentwise product with a vector is
# written as a product with the corresponding diagonal matrix, C = diag(c),
# e.g., 'b{σ}.C{σ,ν}.c{σ,μ}' denotes (b^{σ})^T diag(c^{σ,ν}) c^{σ,μ}, and
# 'b{σ}.diag(A{σ,ν}.c{ν,μ}).A{σ,λ}.c{λ,κ}' denotes
# (b^{σ})^T diag(A^{σ,ν} c^{ν,μ}) A^{σ,λ} c^{λ,κ}.  Labels may instead be
# generated in LaTeX, using the macros of the lecture notes.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

# general imports
import os
import sys
from fractions import Fraction
from math import comb
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from utilities.rooted_trees import rooted_trees, tree_density

# placeholder color names, in the order that they are assigned to vertices
text_colors = ['σ', 'ν', 'μ', 'λ', 'κ', 'ρ', 'τ', 'ω']
latex_colors = ['\\sigma', '\\nu', '\\mu', '\\lambda', '\\kappa', '\\rho', '\\tau', '\\omega']

def condition_label(t, internal=False, latex=False):
    ''' Usage: label = condition_label(t, internal, latex)

        Inputs:
          t is a rooted tree
          internal is optional, indicating that the method is internally
             consistent, c^{sigma,nu} = c^{sigma} (default False)
          latex is optional, generating the label in LaTeX (default False)

        Outputs:
          label is the left-hand side of the order condition for t, with
             placeholder colors assigned to the vertices in the order in
             which they appear in the label
    '''
    names = latex_colors if latex else text_colors
    count = [0]
    def new_color():
        name = names[count[0]]
        count[0] += 1
        return name
    def sup(colors):
        if latex:
            return '^{\\{' + ','.join(colors) + '\\}}'
        return '{' + ','.join(colors) + '}'
    def vector(children, x):
        # leaves contribute c (listed first), and all other subtrees contribute
        # A times the vector for their own children
        factors = []
        for child in children:
            if (len(child) == 0):
                colors = (x,) if internal else (x, new_color())
                factors.append(('leaf', sup(colors), ''))
        for child in children:
            if (len(child) > 0):
                y = new_color()
                factors.append(('term', sup((x, y)), vector(child, y)))
        if latex:
            label = ''
            for k, (kind, s, v) in enumerate(factors):
                last = (k == len(factors)-1)
                if (kind == 'leaf'):
                    label += ('\\cvec' if last else 'C') + s
                elif last:
                    label += 'A' + s + v
                else:
                    label += '\\operatorname{diag}\\(A' + s + v + '\\)'
            return label
        labels = []
        for k, (kind, s, v) in enumerate(factors):
            last = (k == len(factors)-1)
            if (kind == 'leaf'):
                labels.append(('c' if last else 'C') + s)
            elif last:
                labels.append('A' + s + '.' + v)
            else:
                labels.append('diag(A' + s + '.' + v + ')')
        return '.'.join(labels)
    x = new_color()
    if latex:
        if (len(t) == 0):
            return '(\\bvec' + sup((x,)) + ')^T\\onevec'
        return '(\\bvec' + sup((x,)) + ')^T' + vector(t, x)
    if (len(t) == 0):
        return 'b' + sup((x,)) + '.1'
    return 'b' + sup((x,)) + '.' + vector(t, x)


def GARK_conditions(order, internal=False, latex=False):
    ''' Usage: conds = GARK_conditions(order, internal, latex)

        Inputs:
          order is the order of the conditions to generate
          internal is optional, indicating that the method is internally
             consistent, c^{sigma,nu} = c^{sigma} (default False)
          latex is optional, generating the labels in LaTeX (default False)

        Outputs:
          conds is a list of tuples (label, rhs), one for each rooted tree
             of the given order, where the condition label = rhs must hold
             for every choice of the placeholder colors in {1,...,M}
    '''
    conds = [(condition_label(t, internal, latex), Fraction(1, tree_density(t)))
             for t in rooted_trees(order)]
    conds.sort(key=lambda cond: (cond[1], cond[0]))
    return conds


def num_colorings(t, M, internal=False, root=True):
    ''' Usage: n = num_colorings(t, M, internal, root)

        Returns the number of distinct colorings of the tree t using M
        colors, i.e., the number of distinct order conditions that t
        generates for an M-partition GARK method.  If internal is True, the
        (non-root) leaves are not colored.  The optional input root indicates
        whether t is a full tree (True, default), or a subtree below the root.
    '''
    if ((len(t) == 0) and internal and (not root)):
        return 1
    n = M
    for child in set(t):
        m = t.count(child)
        n *= comb(num_colorings(child, M, internal, False) + m - 1, m)
    return n


def num_conditions(order, M, internal=False):
    ''' Usage: total, coupling = num_conditions(order, M, internal)

        Returns the total number of distinct order conditions of the given
        order for an M-partition GARK method, and the number of these that
        are coupling conditions (i.e., that are not an order condition of a
        single base method).
    '''
    total = 0
    coupling = 0
    for t in rooted_trees(order):
        n = num_colorings(t, M, internal)
        total += n
        coupling += n - M
    return total, coupling


def count_polynomial(order, internal=False, coupling=False):
    ''' Usage: coeffs = count_polynomial(order, internal, coupling)

        Returns the coefficients [a_0, a_1, ..., a_order] (as Fractions) of
        the polynomial in M that gives the number of distinct order
        conditions (or, if coupling is True, coupling conditions) of the
        given order for an M-partition GARK method.  Since this number is a
        polynomial of degree at most order, it is found by interpolating its
        values at M = 0, 1, ..., order.
    '''
    Ms = list(range(order+1))
    vals = [Fraction(num_conditions(order, M, internal)[1 if coupling else 0]) for M in Ms]
    coeffs = [Fraction(0)] * (order+1)
    for i in range(order+1):
        # Lagrange basis polynomial for node Ms[i], as a coefficient list
        basis = [Fraction(1)]
        denom = Fraction(1)
        for j in range(order+1):
            if (j != i):
                basis = [Fraction(0)] + basis
                for k in range(len(basis)-1):
                    basis[k] -= Ms[j]*basis[k+1]
                denom *= Ms[i] - Ms[j]
        for k in range(order+1):
            coeffs[k] += vals[i]*basis[k]/denom
    return coeffs


def polynomial_string(coeffs):
    ''' Usage: s = polynomial_string(coeffs)

        Returns a string for the polynomial in M with the coefficients
        [a_0, a_1, ...], listing the highest powers first.
    '''
    s = ''
    for k in range(len(coeffs)-1, -1, -1):
        a = coeffs[k]
        if (a == 0):
            continue
        sign = ' - ' if (a < 0) else ' + '
        if (s == ''):
            sign = '-' if (a < 0) else ''
        a = abs(a)
        mono = '' if (k == 0) else ('M' if (k == 1) else 'M^%i' % (k))
        if (mono == ''):
            term = str(a)
        elif (a == 1):
            term = mono
        else:
            term = str(a) + ' ' + mono
        s += sign + term
    return s if (s != '') else '0'


# main routine
if __name__ == '__main__':

    # highest order to consider, and specific numbers of partitions to count
    maxorder = 6
    Mvals = [2, 3]

    cases = [('general', False), ('internally consistent', True)]

    # print the conditions themselves
    for name, internal in cases:
        print("\nGARK order conditions (%s), for all colors in {1,...,M}:" % (name))
        for q in range(1,maxorder+1):
            print("  order %i:" % (q))
            for label, rhs in GARK_conditions(q, internal):
                print("    %s = %s" % (label, rhs))

    # count the distinct and coupling conditions as polynomials in M
    for name, internal in cases:
        print("\nNumber of distinct conditions (%s), as polynomials in M:" % (name))
        for q in range(1,maxorder+1):
            print("  order %i:  total = %s,  coupling = %s" %
                  (q, polynomial_string(count_polynomial(q, internal)),
                   polynomial_string(count_polynomial(q, internal, True))))

    # count the coupling conditions for specific numbers of partitions
    print("\nNumber of coupling conditions at each order:")
    print("  order" + "".join("%15s" % ("M=%i" % (M)) for M in Mvals)
          + "".join("%15s" % ("M=%i, int." % (M)) for M in Mvals))
    for q in range(1,maxorder+1):
        counts = [num_conditions(q, M)[1] for M in Mvals] \
                 + [num_conditions(q, M, True)[1] for M in Mvals]
        print("  %5i" % (q) + "".join("%15i" % (n) for n in counts))
