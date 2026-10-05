# ImEx_LMM.py
#
# Fixed-stepsize implicit-explicit linear multistep class implementation file.
#
# Also contains functions to return ImEx-LMM coefficients
# of order 2.
#
# Class to perform fixed-stepsize time evolution of the IVP
#      y' = fe(t,y) + fi(t,y),  t in [t0, Tf],  y(t0) = y0
# using an implicit-explicit linear multistep (LMM) time stepping method.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import sys
sys.path.append('..')
from shared.ImplicitSolver import *

class ImEx_LMM:
    r"""
    Fixed stepsize implicit-explicit linear multistep class

    The six required arguments when constructing an ImEx linear
    multistep object are functions for the explicit and implicit
    portions of the IVP right-hand side, an implicit solver to use,
    and the LMM coefficients:
        fe = ODE RHS function for the explicit portion, with calling syntax fe(t,y).
        fi = ODE RHS function for the implicit portion, with calling syntax fi(t,y).
        sol = algebraic solver object to use [ImplicitSolver]; this should
              be constructed using the Jacobian of fi
        alpha = LMM coefficients on previous solution values
        beta = LMM coefficients on previous implicit RHS values
        gamma = LMM coefficients on previous explicit RHS values (gamma[0] must be 0)
        h = (optional) input with stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    Note that the ImEx LMM has the form:
       \sum_{j=0}^{k-1} \alpha_j y_{n+1-j} = h\sum_{j=0}^{k-1} \beta_j fi_{n+1-j}
                                           + h\sum_{j=1}^{k-1} \gamma_j fe_{n+1-j},
    for computing each internal step, for an ODE IVP of the form
       y' = fe(t,y) + fi(t,y), t in tspan,
       y(t0) = y0.
    """
    def __init__(self, fe, fi, sol, alpha, beta, gamma, h=0.0):
        # required inputs
        self.fe = fe
        self.fi = fi
        self.sol = sol
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # optional inputs
        self.h = h
        # internal data
        self.steps = 0
        self.k = alpha.size

        # verify that input LMM coefficients are valid
        if (abs(alpha[0]) == 0):
            raise ValueError("ImEx_LMM ERROR: alpha[0] = ", alpha[0], " (should be nonzero)")
        if (abs(beta[0]) == 0):
            raise ValueError("ImEx_LMM ERROR: beta[0] = ", beta[0], " (should be nonzero)")
        if (gamma[0] != 0):
            raise ValueError("ImEx_LMM ERROR: gamma[0] = ", gamma[0], " (should be zero)")
        if (beta.size != self.k):
            raise ValueError("ImEx_LMM ERROR: alpha and beta do not have the same length, (",
                             alpha.size, " != ", beta.size, ")")
        if (gamma.size != self.k):
            raise ValueError("ImEx_LMM ERROR: alpha and gamma do not have the same length, (",
                             alpha.size, " != ", gamma.size, ")")

    def imex_lmm_step(self, t, args=()):
        """
        Usage: t, success = imex_lmm_step(t, args)

        Utility routine to take a single ImEx LMM time step,
        where the input `t` is overwritten by the updated value.
        args is used for optional parameters of the RHS.
        If success==True then the step succeeded; otherwise it failed.
        """

        # create LMM residual and Jacobian solver for this step
        t += self.h
        self.data = (self.h * self.beta[1] / self.alpha[0]) * self.fiprev[-1] \
            + (self.h * self.gamma[1] / self.alpha[0]) * self.feprev[-1] \
            - (self.alpha[1] / self.alpha[0]) * self.yprev[-1]
        for i in range(2, self.k):
            self.data += (self.h * self.beta[i] / self.alpha[0]) * self.fiprev[-i] \
                + (self.h * self.gamma[i] / self.alpha[0]) * self.feprev[-i] \
                - (self.alpha[i] / self.alpha[0]) * self.yprev[-i]

        # create implicit residual and Jacobian solver for this step
        F = lambda ynew: ynew - self.data - (self.h * self.beta[0] / self.alpha[0]) \
            * self.fi(t, ynew, *args)
        self.sol.setup_linear_solver(t, -self.h * self.beta[0] / self.alpha[0], args)

        # perform implicit solve, and return on solver failure
        y, iters, success = self.sol.solve(F, self.yprev[-1])
        if (not success):
            return t, False

        # add current solution and RHS values to queue, and remove oldest solution and RHS values
        self.yprev.pop(0)
        self.yprev.append(y)
        self.feprev.pop(0)
        self.feprev.append(self.fe(t, y, *args))
        self.fiprev.pop(0)
        self.fiprev.append(self.fi(t, y, *args))
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

        The fixed-step implicit-explicit linear multistep evolution routine.

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
            raise ValueError("ERROR: ImEx_LMM::Evolve called without specifying a nonzero step size")

        # determine the direction of integration from tspan (tdir = 1 forward in
        # time, tdir = -1 backward), require tspan to be monotone in that
        # direction, and sign the internal step size to match it
        if (tspan[-1] >= tspan[0]):
            tdir = 1.0
        else:
            tdir = -1.0
        if (np.any(tdir*np.diff(tspan) < 0)):
            raise ValueError("ERROR: ImEx_LMM::Evolve requires monotone tspan values")
        self.h = tdir*abs(self.h)

        # verify that tspan values are separated by multiples of h
        for n in range(tspan.size-1):
            hn = tspan[n+1]-tspan[n]
            if (abs(round(hn/self.h) - (hn/self.h)) > 100*np.sqrt(np.finfo(self.h).eps)*abs(self.h)):
                raise ValueError("input values in tspan (%e,%e) are not separated by a multiple of h = %e" % (tspan[n],tspan[n+1],self.h))

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
        self.feprev = []
        self.fiprev = []
        self.yprev = []
        for i in range(self.k-1):
            self.yprev.append(y0[i,:])
            self.feprev.append(self.fe(tspan[0]-(self.k-2-i)*self.h, y0[i,:], *args))
            self.fiprev.append(self.fi(tspan[0]-(self.k-2-i)*self.h, y0[i,:], *args))

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required
            N = int(round((tspan[iout]-tspan[iout-1])/self.h))

            # reset "current" t that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform LMM update
                t, success = self.imex_lmm_step(t, args)
                if (not success):
                    print("imex_lmm error in time step at t =", t)
                    return Y, False

            # store current result in output array
            Y[iout,:] = self.yprev[-1]

        # return with "success" flag
        return Y, True

def SBDF2():
    """
    Usage: alphas, betas, gammas, p = SBDF2()

    Utility routine to return the 2nd order semi-implicit BDF (SBDF-2)
    ImEx LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous implicit RHS values
             gammas holds the LMM coefficients on previous explicit RHS values
             p holds the LMM method order
    """
    alphas = np.array([1, -4.0/3.0, 1.0/3.0], dtype=float)
    betas = np.array([2.0/3.0, 0, 0], dtype=float)
    gammas = np.array([0, 4.0/3.0, -2.0/3.0], dtype=float)
    p = 2
    return alphas, betas, gammas, p

def CNAB():
    """
    Usage: alphas, betas, gammas, p = CNAB()

    Utility routine to return the 2nd order Crank-Nicolson, Adams-Bashforth
    (CNAB) ImEx LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous implicit RHS values
             gammas holds the LMM coefficients on previous explicit RHS values
             p holds the LMM method order
    """
    alphas = np.array([1, -1, 0], dtype=float)
    betas = np.array([0.5, 0.5, 0], dtype=float)
    gammas = np.array([0, 1.5, -0.5], dtype=float)
    p = 2
    return alphas, betas, gammas, p

def MCNAB():
    """
    Usage: alphas, betas, gammas, p = MCNAB()

    Utility routine to return the 2nd order modified Crank-Nicolson,
    Adams-Bashforth (MCNAB) ImEx LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous implicit RHS values
             gammas holds the LMM coefficients on previous explicit RHS values
             p holds the LMM method order
    """
    alphas = np.array([1, -1, 0], dtype=float)
    betas = np.array([9.0/16.0, 6.0/16.0, 1.0/16.0], dtype=float)
    gammas = np.array([0, 1.5, -0.5], dtype=float)
    p = 2
    return alphas, betas, gammas, p

def CNLF():
    """
    Usage: alphas, betas, gammas, p = CNLF()

    Utility routine to return the 2nd order Crank-Nicolson, leapfrog
    (CNLF) ImEx LMM coefficients.

    Outputs: alphas holds the LMM coefficients on previous solution values
             betas holds the LMM coefficients on previous implicit RHS values
             gammas holds the LMM coefficients on previous explicit RHS values
             p holds the LMM method order
    """
    alphas = np.array([1, 0, -1], dtype=float)
    betas = np.array([1, 0, 1], dtype=float)
    gammas = np.array([0, 2, 0], dtype=float)
    p = 2
    return alphas, betas, gammas, p
