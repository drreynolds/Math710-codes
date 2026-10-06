#!/usr/bin/env python
# Functions to enumerate the (uncolored) rooted trees of a given order, and to
# compute their order, density, and bracket notation.  Includes a simple
# "main" that uses these functions to list the trees of orders 1 through 5.
#
# Each tree is stored as the sorted tuple of its subtrees, mirroring the
# bracket notation [t_1,...,t_m] of the lecture notes, so that a single vertex
# is the empty tuple ().  Tuples (unlike lists) may be stored in sets and
# sorted, which lets us discard duplicate trees and list each tree, and the
# subtrees of each vertex, in a fixed order.  Use tree_string to display a
# tree in bracket notation.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

# general imports
from functools import lru_cache
from itertools import combinations_with_replacement

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
def rooted_trees(n):
    ''' Usage: trees = rooted_trees(n)

        Returns a sorted tuple of the distinct (uncolored) rooted trees with
        n vertices, each stored as the sorted tuple of its subtrees.
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
            childlists = [a + b for a in childlists
                          for b in combinations_with_replacement(rooted_trees(k), m)]
        for children in childlists:
            trees.add(tuple(sorted(children)))
    return tuple(sorted(trees))


def tree_order(t):
    ''' Usage: n = tree_order(t)

        Returns the number of vertices in the tree t.
    '''
    return 1 + sum(tree_order(child) for child in t)


def tree_density(t):
    ''' Usage: gamma = tree_density(t)

        Returns the density gamma(t) of the tree t, so that the order
        condition for t is Phi(t) = 1/gamma(t).
    '''
    gamma = tree_order(t)
    for child in t:
        gamma *= tree_density(child)
    return gamma


def tree_string(t):
    ''' Usage: s = tree_string(t)

        Returns the tree t in bracket notation, e.g., '[•,[•]]', where '•'
        denotes a single vertex.
    '''
    if (len(t) == 0):
        return '•'
    return '[' + ','.join(tree_string(child) for child in t) + ']'


if __name__ == '__main__':

    # list the trees of orders 1 through 5 in bracket notation, together with
    # their densities
    for q in range(1, 6):
        trees = rooted_trees(q)
        print("\nRooted trees of order %i (count = %i):" % (q, len(trees)))
        for t in trees:
            print("  %-16s gamma = %i" % (tree_string(t), tree_density(t)))
