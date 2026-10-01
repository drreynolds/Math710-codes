#!/usr/bin/env python
# Functions to generate the order conditions (including the coupling
# conditions) for two-component additive Runge--Kutta (ARK) methods, of any
# order, by enumerating bicolored rooted trees.  Optionally applies the
# simplifying assumptions bE = bI and/or cE = cI, under which many of the
# conditions coincide.  The conditions may also be written in a condensed
# form, once per rooted tree, using placeholder colors (nu, mu, lambda, ...)
# for its vertices, where each condition must hold for every choice of these
# colors in {E,I}.  Includes a simple "main" that uses these functions to count
# the coupling conditions at each order, and to print the condensed conditions.
#
# Each node of a tree carries a color, E or I.  The root contributes the
# solution weights b of its color, each other node contributes the matrix A of
# its color, and each leaf therefore contributes the abscissae c = A*1 of its
# color.  If bE = bI, the color of the root is irrelevant; if cE = cI, the
# colors of the (non-root) leaves are irrelevant.
#
# In condition labels, '.' denotes a matrix product, e.g., 'bE.AI.cE' denotes
# (bE)^T AI cE.  A componentwise product with a vector is written as a product
# with the corresponding diagonal matrix, C = diag(c), e.g., 'bI.CE.cI' denotes
# (bI)^T diag(cE) cI, and 'bE.diag(AE.cI).AI.cE' denotes (bE)^T diag(AE cI) AI cE.
# A label without an E or I denotes a factor whose color is irrelevant under
# the simplifying assumptions.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

# general imports
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement
from GARK_colored_trees import rooted_trees
from GARK_colored_trees import tree_density as rooted_tree_density

# placeholder color names, in the order that they are assigned to vertices
placeholders = ['ν', 'μ', 'λ', 'σ', 'κ', 'ρ', 'τ', 'ω']

def partitions(n, maxpart=None):
    ''' Usage: for part in partitions(n, maxpart):

        Generates the partitions of the integer n into parts no larger than
        maxpart (default n), as non-increasing tuples.
    '''
    if (maxpart is None):
        maxpart = n
    if (n == 0):
        yield ()
        return
    for k in range(min(n, maxpart), 0, -1):
        for rest in partitions(n-k, k):
            yield (k,) + rest


@lru_cache(None)
def bicolored_trees(n, root=True, sameb=False, samec=False):
    ''' Usage: trees = bicolored_trees(n, root, sameb, samec)

        Inputs:
          n is the number of nodes in each tree
          root is optional, indicating whether these trees are the full
             trees (True, default), or subtrees below the root (False)
          sameb is optional, indicating that bE = bI (default False)
          samec is optional, indicating that cE = cI (default False)

        Outputs:
          trees is a frozenset of the distinct bicolored rooted trees with n
             nodes, each stored as a tuple (color, children), where color is
             'E', 'I', or '' (when the color is irrelevant), and children is
             a sorted tuple of subtrees
    '''
    trees = set()
    for part in partitions(n-1):
        # count the number of children of each size
        sizes = {}
        for k in part:
            sizes[k] = sizes.get(k, 0) + 1
        # form every multiset of subtrees having these sizes
        childlists = [()]
        for k, m in sizes.items():
            subtrees = sorted(bicolored_trees(k, False, sameb, samec))
            childlists = [a + b for a in childlists
                          for b in combinations_with_replacement(subtrees, m)]
        # color the root of each resulting tree
        for children in childlists:
            children = tuple(sorted(children))
            leaf = (len(children) == 0)
            if ((root and sameb) or ((not root) and leaf and samec)):
                colors = ('',)
            else:
                colors = ('E', 'I')
            for color in colors:
                trees.add((color, children))
    return frozenset(trees)


def tree_colors(t):
    ''' Usage: colors = tree_colors(t)

        Returns the set of relevant colors ('E' and/or 'I') used in the tree t.
    '''
    colors = {t[0]} - {''}
    for child in t[1]:
        colors |= tree_colors(child)
    return colors


def tree_order(t):
    ''' Usage: n = tree_order(t)

        Returns the number of nodes in the tree t.
    '''
    return 1 + sum(tree_order(child) for child in t[1])


def tree_density(t):
    ''' Usage: gamma = tree_density(t)

        Returns the density gamma(t) of the tree t, so that the order
        condition for t is Phi(t) = 1/gamma(t).
    '''
    gamma = tree_order(t)
    for child in t[1]:
        gamma *= tree_density(child)
    return gamma


def vector_label(children):
    ''' Usage: label = vector_label(children)

        Returns the label for the vector formed by the elementwise product of
        the terms contributed by each child subtree.  Leaves contribute c
        (listed first), and all other subtrees contribute A times the vector
        for their own children.  Every term but the last multiplies the terms
        to its right elementwise, and so is written as a diagonal matrix,
        C = diag(c) or diag(A...).
    '''
    leaves = [t[0] for t in children if (len(t[1]) == 0)]
    terms = [f'A{t[0]}.' + vector_label(t[1]) for t in children if (len(t[1]) > 0)]
    factors = ['C' + x for x in leaves] + ['diag(' + term + ')' for term in terms]
    factors[-1] = terms[-1] if (len(terms) > 0) else 'c' + leaves[-1]
    return '.'.join(factors)


def tree_label(t):
    ''' Usage: label = tree_label(t)

        Returns the label for the order condition corresponding to the full
        tree t.
    '''
    if (len(t[1]) == 0):
        return f'b{t[0]}.1'
    return f'b{t[0]}.' + vector_label(t[1])


def ARK_conditions(order, sameb=False, samec=False):
    ''' Usage: conds = ARK_conditions(order, sameb, samec)

        Inputs:
          order is the order of the conditions to generate
          sameb is optional, indicating that bE = bI (default False)
          samec is optional, indicating that cE = cI (default False)

        Outputs:
          conds is a list of tuples (label, rhs, coupling), one for each
             distinct order condition of the given order, where the
             condition is label = rhs, and coupling is True for coupling
             conditions and False for the order conditions of a single
             table
    '''
    conds = []
    for t in bicolored_trees(order, True, sameb, samec):
        coupling = (len(tree_colors(t)) > 1)
        conds.append((tree_label(t), Fraction(1, tree_density(t)), coupling))
    conds.sort(key=lambda cond: (cond[1], cond[0]))
    return conds


def condensed_label(t, sameb=False, samec=False):
    ''' Usage: label = condensed_label(t, sameb, samec)

        Inputs:
          t is an (uncolored) rooted tree, from rooted_trees
          sameb is optional, indicating that bE = bI (default False)
          samec is optional, indicating that cE = cI (default False)

        Outputs:
          label is the left-hand side of the condensed order condition for
             t, with placeholder colors assigned to the vertices in the
             order in which they appear in the label, and omitted for the
             vertices whose colors are irrelevant
    '''
    count = [0]
    def new_color():
        name = placeholders[count[0]]
        count[0] += 1
        return name
    def vector(children):
        # leaves contribute c (listed first), and all other subtrees contribute
        # A times the vector for their own children
        leaves = []
        for child in children:
            if (len(child) == 0):
                leaves.append('' if samec else new_color())
        terms = []
        for child in children:
            if (len(child) > 0):
                y = new_color()
                terms.append('A' + y + '.' + vector(child))
        # every factor but the last is written as a diagonal matrix
        factors = ['C' + x for x in leaves] + ['diag(' + term + ')' for term in terms]
        factors[-1] = terms[-1] if (len(terms) > 0) else 'c' + leaves[-1]
        return '.'.join(factors)
    x = '' if sameb else new_color()
    if (len(t) == 0):
        return 'b' + x + '.1'
    return 'b' + x + '.' + vector(t)


def ARK_condensed_conditions(order, sameb=False, samec=False):
    ''' Usage: conds = ARK_condensed_conditions(order, sameb, samec)

        Inputs:
          order is the order of the conditions to generate
          sameb is optional, indicating that bE = bI (default False)
          samec is optional, indicating that cE = cI (default False)

        Outputs:
          conds is a list of tuples (label, rhs), one for each rooted tree
             of the given order, where the condition label = rhs must hold
             for every choice of the placeholder colors in {E,I}
    '''
    conds = [(condensed_label(t, sameb, samec), Fraction(1, rooted_tree_density(t)))
             for t in rooted_trees(order)]
    conds.sort(key=lambda cond: (cond[1], cond[0]))
    return conds


# main routine
if __name__ == '__main__':

    # highest orders for the count table and for printing the conditions
    maxorder_count = 6
    maxorder_print = 6

    cases = [('general', False, False), ('bE=bI', True, False),
             ('cE=cI', False, True), ('bE=bI, cE=cI', True, True)]

    # count the coupling conditions at each order, under each set of assumptions
    print("\nNumber of coupling conditions at each order:")
    print("  order" + "".join("%15s" % (name) for name, _, _ in cases))
    for q in range(1,maxorder_count+1):
        counts = [sum(1 for cond in ARK_conditions(q, sameb, samec) if cond[2])
                  for _, sameb, samec in cases]
        print("  %5i" % (q) + "".join("%15i" % (n) for n in counts))

    # print the condensed conditions themselves
    for name, sameb, samec in cases:
        print("\nOrder conditions (%s), for all colors in {E,I}:" % (name))
        for q in range(1,maxorder_print+1):
            print("  order %i:" % (q))
            for label, rhs in ARK_condensed_conditions(q, sameb, samec):
                print("    %s = %s" % (label, rhs))
