# Implicit_LMM.py
#
# Fixed-stepsize implicit linear multistep class implementation file.
#
# Also contains functions to return Adams-Moulton and BDF LMM coefficients
# of orders 1-4.
#
# Class to perform fixed-stepsize time evolution of the IVP
#      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
# using an implicit linear multistep (LMM) time stepping method.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import sys
sys.path.append('..')
from shared.ImplicitSolver import *

class Implicit_LMM:
    r"""
    Fixed stepsize implicit linear multistep class

    The four required arguments when constructing an implicit linear
    multistep object are a function for the IVP right-hand side, an
    implicit solver to use, and the LMM coefficients:
        f = ODE RHS function with calling syntax f(t,y).
        sol = algebraic solver object to use [ImplicitSolver]
        alpha = LMM coefficients on previous solution values
        beta = LMM coefficients on previous RHS values
        h = (optional) input with stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    Note that the LMM has the form:
       \sum_{j=0}^{k-1} \alpha_j y_{n+1-j} = h\sum_{j=0}^{k-1} \beta_j f_{n+1-j},
    for computing each internal step, for an ODE IVP of the form
       y' = f(t,y), t in tspan,
       y(t0) = y0.
    """
    def __init__(self, f, sol, alpha, beta, h=0.0):
        # required inputs
        self.f = f
        self.sol = sol
        self.alpha = alpha
        self.beta = beta
        # optional inputs
        self.h = h
        # internal data
        self.steps = 0
        self.k = alpha.size

        # verify that input LMM coefficients are valid
        if (abs(alpha[0]) == 0):
            raise ValueError("Implicit_LMM ERROR: alpha[0] = ", alpha[0], " (should be nonzero)")
        if (beta.size != self.k):
            raise ValueError("Implicit_LMM ERROR: alpha and beta do not have the same length, (",
                             alpha.size, " != ", beta.size, ")")

    def implicit_lmm_step(self, t, args=()):
        """
        Usage: t, success = implicit_lmm_step(t, args)

        Utility routine to take a single implicit LMM time step,
        where the input `t` is overwritten by the updated value.
        args is used for optional parameters of the RHS.
        If success==True then the step succeeded; otherwise it failed.
        """

        # create LMM residual and Jacobian solver for this step
        t += self.h
        self.data = (self.h * self.beta[1] / self.alpha[0]) * self.fprev[-1] \
            - (self.alpha[1] / self.alpha[0]) * self.yprev[-1]
        for i in range(2, self.k):
            self.data += (self.h * self.beta[i] / self.alpha[0]) * self.fprev[-i] \
                - (self.alpha[i] / self.alpha[0]) * self.yprev[-i]

        # create implicit residual and Jacobian solver for this step
        F = lambda ynew: ynew - self.data - (self.h * self.beta[0] / self.alpha[0]) \
            * self.f(t, ynew, *args)
        self.sol.setup_linear_solver(t, -self.h * self.beta[0] / self.alpha[0], args)

        # perform implicit solve, and return on solver failure
        y, iters, success = self.sol.solve(F, self.yprev[-1])
        if (not success):
            return t, False

        # add current solution and RHS to queue, and remove oldest solution and RHS
        self.yprev.pop(0)
        self.yprev.append(y)
        self.fprev.pop(0)
        self.fprev.append(self.f(t, y, *args))
        self.steps += 1
        return t, True

    def reset(self):
        """ Resets the accumulated number of steps """
        self.steps = 0

    def get_num_steps(self):
        """ Returns the accumulated number of steps """
        return self.steps

    def get_num_solves(self):
        """ Returns the accumulated number of implicit solves """
        return self.steps

    def Evolve(self, tspan, y0, h=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, h, args)

        The fixed-step implicit linear multistep evolution routine.

        Note: this requires that y0 has separate rows containing
        sufficiently accurate "initial" values for all previous LMM steps.

        Inputs:  tspan holds the current time interval, [t0, tf], including any
                     intermediate times when the solution is desired, i.e.
                     [t0, t1, ..., tf]; these may decrease (to integrate
                     backward in time), but must be monotone
                 y0 holds the initial conditions [nd-array, shape(k-1,n)],
                     sorted as [y0(t0-(k-2)*h), ... y0(t0-h), y0(t0)], where
                     h is signed in the direction of integration (h < 0 when
                     tspan is decreasing)
                 h optionally holds the requested step size magnitude (if it
                     is not provided then the stored value will be used)
                 args holds optional equation parameters used when evaluating
                     the RHS.
        Outputs: Y holds the computed solution at all tspan values,
                     [y(t0), y(t1), ..., y(tf)]
                 success = True if the solver traversed the interval,
                     false if an integration step failed [bool]
        """

        # set time step for evolution based on input-vs-stored value
        if (h != 0.0):
            self.h = h

        # raise error if step size was never set
        if (self.h == 0.0):
            raise ValueError("ERROR: Implicit_LMM::Evolve called without specifying a nonzero step size")

        # determine the direction of integration from tspan (tdir = 1 forward in
        # time, tdir = -1 backward), require tspan to be monotone in that
        # direction, and sign the internal step size to match it
        if (tspan[-1] >= tspan[0]):
            tdir = 1.0
        else:
            tdir = -1.0
        if (np.any(tdir*np.diff(tspan) < 0)):
            raise ValueError("ERROR: Implicit_LMM::Evolve requires monotone tspan values")
        self.h = tdir*abs(self.h)

        # verify that tspan values are separated by multiples of h
        for n in range(tspan.size-1):
            hn = tspan[n+1]-tspan[n]
            if (abs(round(hn/self.h) - (hn/self.h)) > 100*np.sqrt(np.finfo(h).eps)*abs(self.h)):
                raise ValueError("input values in tspan (%e,%e) are not separated by a multiple of h = %e" % (tspan[n],tspan[n+1],h))

        # verify that a sufficient set of initial conditions have been supplied
        if (np.shape(y0)[0] < (self.k-1)):
            raise ValueError("insufficient initial conditions provided, ",
                             np.shape(y0)[0], " < ", self.k-1)

        # initialize outputs, and set first entry corresponding to initial condition
        t = np.zeros(tspan.size)
        Y = np.zeros((tspan.size,y0.shape[1]))
        Y[0,:] = y0[-1,:]

        # initialize internal solution-vector-sized data
        self.data = np.copy(y0[-1,:])
        self.fprev = []
        self.yprev = []
        for i in range(self.k-1):
            self.yprev.append(y0[i,:])
            self.fprev.append(self.f(tspan[0]-(self.k-2-i)*self.h, y0[i,:], *args))

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required
            N = int(round((tspan[iout]-tspan[iout-1])/self.h))

            # reset "current" t that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform LMM update
                t, success = self.implicit_lmm_step(t, args)
                if (not success):
                    print("implicit_lmm error in time step at t =", t)
                    return Y, False

            # store current result in output array
            Y[iout,:] = self.yprev[-1]

        # return with "success" flag
        return Y, True

def AdamsMoulton1():
    """
    Usage: alphas, betas, p = AdamsMoulton1()

    Utility routine to return the 1st order Adams Moulton LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous RHS values
             p holds the LMM method order
    """
    alphas = np.array([1, -1], dtype=float)
    betas = np.array([1, 0], dtype=float)
    p = 1
    return alphas, betas, p

def AdamsMoulton2():
    """
    Usage: alphas, betas, p = AdamsMoulton2()

    Utility routine to return the 2nd order Adams Moulton LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous RHS values
             p holds the LMM method order
    """
    alphas = np.array([1, -1], dtype=float)
    betas = np.array([0.5, 0.5], dtype=float)
    p = 2
    return alphas, betas, p

def AdamsMoulton3():
    """
    Usage: alphas, betas, p = AdamsMoulton3()

    Utility routine to return the 3rd order Adams Moulton LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous RHS values
             p holds the LMM method order
    """
    alphas = np.array([1, -1, 0], dtype=float)
    betas = np.array([5.0/12.0, 8.0/12.0, -1.0/12.0], dtype=float)
    p = 3
    return alphas, betas, p

def AdamsMoulton4():
    """
    Usage: alphas, betas, p = AdamsMoulton4()

    Utility routine to return the 4th order Adams Moulton LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous RHS values
             p holds the LMM method order
    """
    alphas = np.array([1, -1, 0, 0], dtype=float)
    betas = np.array([9.0/24.0, 19.0/24.0, -5.0/24.0, 1.0/24.0], dtype=float)
    p = 4
    return alphas, betas, p

def BDF1():
    """
    Usage: alphas, betas, p = BDF1()

    Utility routine to return the 1st order BDF LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous RHS values
             p holds the LMM method order
    """
    alphas = np.array([1, -1], dtype=float)
    betas = np.array([1, 0], dtype=float)
    p = 1
    return alphas, betas, p

def BDF2():
    """
    Usage: alphas, betas, p = BDF2()

    Utility routine to return the 2nd order BDF LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous RHS values
             p holds the LMM method order
    """
    alphas = np.array([1, -4.0/3.0, 1.0/3.0], dtype=float)
    betas = np.array([2.0/3.0, 0, 0], dtype=float)
    p = 2
    return alphas, betas, p

def BDF3():
    """
    Usage: alphas, betas, p = BDF3()

    Utility routine to return the 3rd order BDF LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous RHS values
             p holds the LMM method order
    """
    alphas = np.array([1, -18.0/11.0, 9.0/11.0, -2.0/11.0], dtype=float)
    betas = np.array([6.0/11.0, 0, 0, 0], dtype=float)
    p = 3
    return alphas, betas, p

def BDF4():
    """
    Usage: alphas, betas, p = BDF4()

    Utility routine to return the 4th order BDF LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous RHS values
             p holds the LMM method order
    """
    alphas = np.array([1.0, -48.0/25.0, 36.0/25.0, -16.0/25.0, 3.0/25.0], dtype=float)
    betas = np.array([12.0/25.0, 0, 0, 0, 0], dtype=float)
    p = 4
    return alphas, betas, p
