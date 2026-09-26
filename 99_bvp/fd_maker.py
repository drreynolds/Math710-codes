#!/usr/bin/env python
#
# fd_maker.py
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

def fd_maker(stencil, deriv):
    r"""
    Usage: coeffs, errorterm = fd_maker(stencil, deriv)

    Utility to compute classical finite difference approximation to a requested derivative
    using a specified set of nodes.  We assume that all nodes are evenly spaced, with a
    spacing of 'h', and that the derivative is requested at the node "0", meaning that the
    location of each node may be uniquely specified by a "stencil" of offsets from the
    derivative location.

    Inputs:  stencil = array of integer offsets from node "0" that
                        will be used in approximation, e.g. [-1, 0, 1]
                        for f(x-h), f(x) and f(x+h)
             deriv = integer specifying the desired derivative

    Outputs: coeffs = row vector of finite-difference coefficients s.t.
                         f^(deriv) \approx \sum coeffs(i)*f(x+stencil(i)*h)
             errorterm = leading error term in derivative approximation
    """

    # imports
    import sympy

    # check for a sufficient number of coefficients
    n = len(stencil)
    if (deriv > n-1):
        raise ValueError("not enough stencil entries for requested derivative")

    # create and solve symbolic linear system for coefficients
    h = sympy.Symbol("h", positive = True)
    A = sympy.zeros(n,n)
    for i in range(n):
        for j in range(n):
            A[i,j] = stencil[j]**i
    fact = h**(-deriv)
    for i in range(deriv):
        fact = fact*(i+1)
    b = sympy.zeros(n,1)
    b[deriv] = fact
    sol = sympy.linsolve((A,b))
    ci = next(iter(sol))
    c = sympy.zeros(n,1)
    for i in range(n):
        c[i] = ci[i]

    # check conditions (up to twice as far along) to find error term
    #   create larger linear system of conditions
    A = sympy.zeros(2*n,n)
    for i in range(2*n):
        for j in range(n):
            A[i,j] = stencil[j]**i
    b = sympy.zeros(2*n,1)
    b[deriv] = fact

    #   determine which equations fail, scale by corresponding factor of h^i/i!
    err = A * c - b
    for i in range(2*n):
        err[i] *= h**i / sympy.factorial(i)

    #   identify the leading nonzero error term and store for output
    for i in range(2*n):
        if (abs(err[i]) > 0):
            errorterm = err[i]
            break

    return c.T, errorterm


if __name__ == '__main__':
    """
    Driver that shows example usage of fd_maker to generate finite-difference
    stencils.
    """

    print("\nCentered 3-point stencil for the first derivative:")
    stencil = [-1, 0, 1]
    coeffs, errorterm = fd_maker(stencil, 1)
    print("  stencil = ", stencil)
    print("  coefficients = ", coeffs)
    print("  error term = ", errorterm)

    print("\nOne-sided (right facing) 6-point stencil for the second derivative:")
    stencil = [0, 1, 2, 3, 4, 5]
    coeffs, errorterm = fd_maker(stencil, 2)
    print("  stencil = ", stencil)
    print("  coefficients = ", coeffs)
    print("  error term = ", errorterm)

    print("\nLopsided 8-point stencil for the third derivative:")
    stencil = [-2, -1, 0, 1, 2, 3, 4, 5]
    coeffs, errorterm = fd_maker(stencil, 3)
    print("  stencil = ", stencil)
    print("  coefficients = ", coeffs)
    print("  error term = ", errorterm)

# end of file
