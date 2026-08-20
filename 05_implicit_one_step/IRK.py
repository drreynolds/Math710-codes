# IRK.py
#
# Fixed-stepsize fully-implicit Runge--Kutta stepper class
# implementation file.
#
# Also contains functions to return specific IRK Butcher tables.
#
# Class to perform fixed-stepsize time evolution of the IVP
#      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
# using a fully-implicit Runge--Kutta (IRK) time stepping
# method.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import sys
sys.path.append('..')
from shared.ImplicitSolver import *

class IRK:
    """
    Fixed stepsize fully-implicit Runge--Kutta class

    The five required arguments when constructing an IRK object are a
    function for the IVP right-hand side, an implicit solver to use,
    and a Butcher table:
        f = ODE RHS function with calling syntax f(t,y).
        sol = algebraic solver object to use [ImplicitSolver]
        A = Runge--Kutta stage coefficients (s*s matrix)
        b = Runge--Kutta solution weights (s array)
        c = Runge--Kutta abcissae (s array).
        h = (optional) input with stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    """
    def __init__(self, f, sol, B, h=0.0):
        # required inputs
        self.f = f
        self.sol = sol
        self.A = B['A']
        self.b = B['b']
        self.c = B['c']

        # optional inputs
        self.h = h

        # internal data
        self.steps = 0
        self.nsol = 0
        self.s = self.c.size

        # check for legal table
        if ((np.size(self.c,0) != self.s) or (np.size(self.A,0) != self.s) or
            (np.size(self.A,1) != self.s)):
            raise ValueError("IRK ERROR: incompatible Butcher table supplied")

    def irk_step(self, t, y, args=()):
        """
        Usage: t, y, success = irk_step(t, y, args)

        Utility routine to take a single fully-implicit RK time step,
        where the inputs (t,y) are overwritten by the updated versions.
        If success==True then the step succeeded; otherwise it failed.
        """
        from scipy.linalg import lu_factor
        from scipy.linalg import lu_solve
        from scipy.sparse import identity
        from scipy.sparse.linalg import LinearOperator
        from scipy.sparse.linalg import gmres
        from scipy.sparse.linalg import factorized

        # define IRK residual function
        s = self.s
        m = y.size
        def F(z):
            # first portion: zi-yold
            resid = np.copy(z)
            for i in range(s):
                resid[m*i:m*(i+1)] -= y
            # second portion: -h*sum[Aij*f(t+cj*h,zj)]
            for j in range(s):
                tj = t + self.c[j] * self.h
                zj = np.array(z[m*j:m*(j+1)])
                self.k[j,:] = self.f(tj, zj, *args)
                for i in range(s):
                    resid[m*i:m*(i+1)] -= self.h * self.A[i,j] * self.k[j,:]
            return resid

        # construct Jacobian solver for this stage
        if (self.sol.solver_type == 'dense'):
            def J(z):
                Jac = np.eye(z.size)
                for j in range(s):
                    tj = t + self.c[j] * self.h
                    zj = np.array(z[m*j:m*(j+1)])
                    Jj = self.sol.f_y(tj, zj, *args)
                    for i in range(s):
                        Jac[m*i:m*(i+1),m*j:m*(j+1)] -= self.h * self.A[i,j] * Jj
                try:
                    lu, piv = lu_factor(Jac)
                except:
                    raise RuntimeError("Dense Jacobian factorization failure")
                Jsolve = lambda b: lu_solve((lu, piv), b)
                return LinearOperator((z.size,z.size), matvec=Jsolve)
        elif (self.sol.solver_type == 'sparse'):
            def J(z):
                Jac = identity(z.size)
                for j in range(s):
                    tj = t + self.c[j] * self.h
                    zj = np.array(z[m*j:m*(j+1)])
                    Jj = self.sol.f_y(tj, zj, *args)
                    for i in range(s):
                        Jac[m*i:m*(i+1),m*j:m*(j+1)] -= self.h * self.A[i,j] * Jj
                try:
                    Jfactored = factorized(Jac)
                except:
                    raise RuntimeError("Sparse Jacobian factorization failure")
                Jsolve = lambda b: Jfactored(b)
                return LinearOperator((z.size,z.size), matvec=Jsolve)
        self.sol.linear_solver = J

        # create initial guess for time-evolved solution
        for i in range(s):
            self.z[m*i:m*(i+1)] = np.copy(y)

        # perform implicit solve, and return on solver failure
        self.z, iters, success = self.sol.solve(F, self.z)
        self.nsol += 1
        if (not success):
            return t, y, False

        # compute updated time step solution
        for i in range(s):
            y += self.h * self.b[i] * self.k[i,:]
        t += self.h
        self.steps += 1
        return t, y, True

    def reset(self):
        """ Resets the accumulated number of steps """
        self.steps = 0
        self.nsol = 0

    def get_num_steps(self):
        """ Returns the accumulated number of steps """
        return self.steps

    def get_num_solves(self):
        """ Returns the accumulated number of implicit solves """
        return self.nsol

    def Evolve(self, tspan, y0, h=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, h, args)

        The fixed-step IRK evolution routine.

        Inputs:  tspan holds the current time interval, [t0, tf], including any
                     intermediate times when the solution is desired, i.e.
                     [t0, t1, ..., tf]
                 y holds the initial condition, y(t0)
                 h optionally holds the requested step size (if it is not
                     provided then the stored value will be used)
                 args holds optional equation parameters used when evaluating
                     the RHS.
        Outputs: Y holds the computed solution at all tspan values,
                     [y(t0), y(t1), ..., y(tf)]
                 success = True if the solver traversed the interval,
                     false if an integration step failed [bool]
        """

        # set time step for evoluation based on input-vs-stored value
        if (h != 0.0):
            self.h = h

        # raise error if step size was never set
        if (self.h == 0.0):
            raise ValueError("ERROR: DIRK::Evolve called without specifying a nonzero step size")

        # verify that tspan values are separated by multiples of h
        for n in range(tspan.size-1):
            hn = tspan[n+1]-tspan[n]
            if (abs(round(hn/self.h) - (hn/self.h)) > 100*np.sqrt(np.finfo(h).eps)*abs(self.h)):
                raise ValueError("input values in tspan (%e,%e) are not separated by a multiple of h = %e" % (tspan[n],tspan[n+1],h))

        # initialize output, and set first entry corresponding to initial condition
        y = y0.copy()
        Y = np.zeros((tspan.size,y0.size))
        Y[0,:] = y

        # initialize internal solution-vector-sized data
        self.k = np.zeros((self.s, y0.size), dtype=float)
        self.z = np.zeros((self.s * y0.size), dtype=float)

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required
            N = int(round((tspan[iout]-tspan[iout-1])/self.h))

            # reset "current" t that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform diagonally-implicit Runge--Kutta update
                t, y, success = self.irk_step(t, y, args)
                if (not success):
                    print("IRK::Evolve error in time step at t =", t)
                    return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with "success" flag
        return Y, True

def RadauIIA2():
    """
    Usage: B = RadauIIA2()

    Utility routine to return the O(h^3) RadauIIA 2-stage IRK table.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array(((5.0/12.0, -1.0/12.0),
                  (9.0/12.0, 3.0/12.0)))
    b = np.array((0.75, 0.25))
    c = np.array((1.0/3.0, 1.0))
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def GaussLegendre2():
    """
    Usage: B = GaussLegendre2()

    Utility routine to return the O(h^4) Gauss-Legendre 2-stage IRK table.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array(((0.25, (3.0-2.0*np.sqrt(3.0))/12.0),
                  ((3.0+2.0*np.sqrt(3.0))/12.0, 0.25)))
    b = np.array((0.5, 0.5))
    c = np.array(((3.0 - np.sqrt(3.0))/6.0, (3.0 + np.sqrt(3.0))/6.0))
    p = 4
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def RadauIIA3():
    """
    Usage: B = RadauIIA3()

    Utility routine to return the O(h^5) RadauIIA 3-stage IRK table.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((( (88.0 - 7.0*np.sqrt(6.0))/360.0,
                    (296.0 - 169.0*np.sqrt(6.0))/1800.0,
                    (-2.0 + 3.0*np.sqrt(6.0))/225.0),
                  ( (296.0 + 169.0*np.sqrt(6.0))/1800.0,
                    (88.0 + 7.0*np.sqrt(6.0))/360.0,
                    (-2.0 - 3.0*np.sqrt(6.0))/225.0),
                  ( (16.0 - np.sqrt(6.0))/36.0,
                    (16.0 + np.sqrt(6.0))/36.0,
                    1.0/9.0)))
    b = np.array(((16.0 - np.sqrt(6.0))/36.0,
                  (16.0 + np.sqrt(6.0))/36.0,
                  1.0/9.0))
    c = np.array(((4.0 - np.sqrt(6.0))/10.0, (4.0 + np.sqrt(6.0))/10.0, 1.0))
    p = 5
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def GaussLegendre3():
    """
    Usage: B = GaussLegendre3()

    Utility routine to return the O(h^6) Gauss-Legendre 3-stage IRK table.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array(( (5.0/36.0, 2.0/9.0 - np.sqrt(15.0)/15.0, 5.0/36.0 - np.sqrt(15.0)/30.0),
                   (5.0/36.0 + np.sqrt(15.0)/24.0, 2.0/9.0, 5.0/36.0 - np.sqrt(15.0)/24.0),
                   (5.0/36.0 + np.sqrt(15.0)/30.0, 2.0/9.0 + np.sqrt(15.0)/15.0, 5.0/36.0) ))
    b = np.array((5.0/18.0, 4.0/9.0, 5.0/18.0))
    c = np.array(((5.0 - np.sqrt(15.0))/10.0, 0.5, (5.0 + np.sqrt(15.0))/10.0))
    p = 6
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def GaussLegendre6():
    """
    Usage: B = GaussLegendre6()

    Utility routine to return the O(h^12) Gauss-Legendre 6-stage IRK table.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((( 0.042831123094792580851996218950605,
                   -0.014763725997197424643891429014278,
                    0.0093250507064777618411400734121424,
                   -0.0056688580494835162182488917046817,
                    0.0028544333150993149102007359161104,
                   -0.00081278017126476782600392067714199 ),
                  ( 0.092673491430378856970823740288243,
                    0.090190393262034655662118827897123,
                   -0.020300102293239581308124404430781,
                    0.010363156240246421640614877198502,
                   -0.0048871929280376802268550750181669,
                    0.001355561055485051944941864725486 ),
                  ( 0.082247922612843859526233540856659,
                    0.19603216233324501065540377853111,
                    0.11697848364317276194496135254516,
                   -0.020482527745656096032756375665715,
                    0.007989991899662334513029865501749,
                   -0.0020756257848663355105554732114538 ),
                  ( 0.087737871974451497214547911112663,
                    0.1723907946244069768112077902925,
                    0.25443949503200161992267908075603,
                    0.11697848364317276194496135254516,
                   -0.015651375809175699331166122736864,
                    0.00341432357674130217775889704455 ),
                  ( 0.084306685134100109759050573175723,
                    0.18526797945210699155109273081241,
                    0.22359381104609910224930782789182,
                    0.2542570695795851051980471095211,
                    0.090190393262034655662118827897123,
                   -0.007011245240793695266831302387034 ),
                  ( 0.086475026360849929529996358578351,
                    0.17752635320896999641403691987814,
                    0.239625825335829040108171596795,
                    0.22463191657986776204878263167818,
                    0.19514451252126673596812908480852,
                    0.042831123094792580851996218950605 )))
    b = np.array((0.085662246189585161703992437901209,
                  0.18038078652406931132423765579425,
                  0.23395696728634552388992270509032,
                  0.23395696728634552388992270509032,
                  0.18038078652406931132423765579425,
                  0.085662246189585161703992437901209))
    c = np.array((0.0337652428984239749709672651079,
                  0.16939530676686775922945571437594,
                  0.38069040695840154764351126459587,
                  0.61930959304159845235648873540413,
                  0.83060469323313224077054428562406,
                  0.9662347571015760250290327348921))
    p = 12
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B
