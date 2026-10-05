# AdaptARK.py
#
# Adaptive-stepsize implicit-explicit additive Runge--Kutta solver class
# implementation file.
#
# Also contains functions to return specific embedded ARK Butcher table pairs.
# The individual explicit and implicit embedded tables are stored in
# AdaptERK.py and AdaptDIRK.py; the routines at the end of this file assemble
# them into compatible pairs.
#
# Class to perform adaptive stepsize time evolution of the IVP
#      y' = fE(t,y) + fI(t,y),  t in [t0, Tf],  y(t0) = y0
# using an embedded implicit-explicit additive Runge--Kutta (ARK) time
# stepping method.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import sys
sys.path.append('..')
from shared.ImplicitSolver import *

class AdaptARK:
    """
    Adaptive implicit-explicit additive Runge--Kutta class

    The six required arguments when constructing an AdaptARK object are
    functions for the explicit and implicit portions of the IVP right-hand
    side, an initial solution vector, an implicit solver to use, and explicit
    and implicit embedded Butcher tables:
        fE = explicit ODE RHS function with calling syntax fE(t,y).
        fI = implicit ODE RHS function with calling syntax fI(t,y).
        y = numpy array with m entries.
        sol = algebraic solver object to use [ImplicitSolver]
        BE = explicit embedded Runge--Kutta Butcher table.
        BI = diagonally-implicit embedded Runge--Kutta Butcher table.
        hE = (optional) known stability step-size limit for fE.
    """
    def __init__(self, fE, fI, y, sol, BE, BI, rtol=1e-3, atol=1e-14, maxit=1e6, bias=1.0, growth=50.0, safety=0.85, hmin=10*np.finfo(float).eps, hE=np.inf, save_step_hist=False):
        # required inputs
        self.fE = fE
        self.fI = fI
        self.sol = sol
        self.AE = BE['A']
        self.bE = BE['b']
        self.cE = BE['c']
        self.dE = BE['d']
        self.AI = BI['A']
        self.bI = BI['b']
        self.cI = BI['c']
        self.dI = BI['d']
        self.minpq = min(BE['p'], BE['q'], BI['p'], BI['q'])

        # optional inputs
        self.rtol = rtol
        self.atol = np.ones(y.size)*atol
        self.maxit = maxit
        self.bias = bias
        self.growth = growth
        self.safety = safety
        self.hmin = hmin
        self.hE = hE

        # internal data
        self.w = np.ones(y.size)
        self.yerr = np.zeros(y.size)
        self.ONEMSM = 1.0 - np.sqrt(np.finfo(float).eps)
        self.ONEPSM = 1.0 + np.sqrt(np.finfo(float).eps)
        self.fails = 0
        self.steps = 0
        self.nsol = 0
        self.save_step_hist = save_step_hist
        self.step_hist = {'t': [], 'h': [], 'err': []}
        self.error_norm = 0.0
        self.h = 0.0
        self.z = np.zeros(y.size)
        self.yt = np.zeros(y.size)
        self.data = np.zeros(y.size)
        self.s = len(self.bE)
        self.kE = np.zeros((self.s, y.size))
        self.kI = np.zeros((self.s, y.size))

        # check for legal tables (including matching numbers of stages)
        if ((np.size(self.cE,0) != self.s) or (np.size(self.cI,0) != self.s) or
            (np.size(self.bI,0) != self.s) or (np.size(self.dE,0) != self.s) or
            (np.size(self.dI,0) != self.s) or (np.size(self.AE,0) != self.s) or
            (np.size(self.AE,1) != self.s) or (np.size(self.AI,0) != self.s) or
            (np.size(self.AI,1) != self.s) or
            (np.linalg.norm(self.bE-self.dE) < 1e-14) or
            (np.linalg.norm(self.bI-self.dI) < 1e-14) or
            (np.linalg.norm(self.AE - np.tril(self.AE,-1), np.inf) > 1e-14) or
            (np.linalg.norm(self.AI - np.tril(self.AI,0), np.inf) > 1e-14)):
            raise ValueError("AdaptARK ERROR: incompatible Butcher tables supplied")
        if ((BE['p'] != BI['p']) or (BE['q'] != BI['q'])):
            raise ValueError("AdaptARK ERROR: Butcher table orders are incompatible")
        if ((self.hE <= 0.0) or (self.hE < self.hmin)):
            raise ValueError("AdaptARK ERROR: illegal explicit stability step-size limit")

    def error_weight(self, y, w):
        """
        Error weight vector utility routine
        """
        for i in range(y.size):
            w[i] = self.bias / (self.atol[i] + self.rtol * np.abs(y[i]))
        return w

    def step(self, t, y, args=()):
        """
        Usage: t, y, success = step(t, y, args)

        Utility routine to take a single implicit-explicit ARK time step,
        where the inputs (t,y) are overwritten by the updated versions.
        args is used for optional parameters of the RHS functions.
        If success==True then the step succeeded; otherwise it failed.
        """

        # loop over stages, computing RHS vectors
        for i in range(self.s):

            # construct "data" for this stage solve
            self.data = np.copy(y)
            for j in range(i):
                self.data += self.h * (self.AE[i,j] * self.kE[j,:]
                                       + self.AI[i,j] * self.kI[j,:])

            # solve the implicit stage (or copy the data for an explicit stage)
            tstageI = t + self.h*self.cI[i]
            if (abs(self.AI[i,i]) > 1e-14):

                # construct implicit residual and Jacobian solver for this stage
                F = lambda zcur: zcur - self.data - self.h * self.AI[i,i] * self.fI(tstageI, zcur, *args)
                self.sol.setup_linear_solver(tstageI, -self.h * self.AI[i,i], args)

                # perform implicit solve, and return on solver failure
                self.z, iters, success = self.sol.solve(F, y)
                self.nsol += 1
                if (not success):
                    return t, y, False
            else:
                self.z = self.data.copy()

            # store both RHS vectors at this stage
            self.kE[i,:] = self.fE(t + self.h*self.cE[i], self.z, *args)
            self.kI[i,:] = self.fI(tstageI, self.z, *args)

        # update time step solution
        for i in range(self.s):
            y += self.h * (self.bE[i] * self.kE[i,:] + self.bI[i] * self.kI[i,:])

        # compute error estimate (and norm), and return
        self.yerr *= 0.0
        for i in range(self.s):
            self.yerr += self.h * ((self.bE[i] - self.dE[i]) * self.kE[i,:]
                                   + (self.bI[i] - self.dI[i]) * self.kI[i,:])
        self.error_norm = max(np.linalg.norm(self.yerr*self.w, np.inf), 1.e-8)
        return t, y, True

    def Evolve(self, tspan, y0, h=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, h, args)

        The adaptive ARK time step evolution routine

        Inputs:  tspan holds the current time interval, [t0, tf], including any
                     intermediate times when the solution is desired, i.e.
                     [t0, t1, ..., tf]; these may decrease (to integrate
                     backward in time), but must be monotone
                 y holds the initial condition, y(t0)
                 h optionally holds the requested initial step size magnitude
                     (if it is not provided then an initial step size will be
                     estimated); this is limited by the explicit stability
                     step-size limit hE
                 args holds optional equation parameters used when evaluating
                     the RHS functions.
        Outputs: Y holds the computed solution at all tspan values,
                     [y(t0), y(t1), ..., y(tf)]
                 success = True if the solver traversed the interval,
                     false if an integration step failed [bool]
        """
        # store input step size
        self.h = h

        # store sizes
        m = len(y0)
        N = len(tspan)-1

        # initialize output
        y = y0.copy()
        Y = np.zeros((N+1, m))
        Y[0,:] = y

        # set current time value
        t = tspan[0]

        # determine the direction of integration from tspan (tdir = 1 forward in
        # time, tdir = -1 backward); the internal step size self.h is kept signed
        # in this direction, so that t + self.h always moves toward tspan[-1]
        if (tspan[-1] >= tspan[0]):
            tdir = 1.0
        else:
            tdir = -1.0

        # check for legal time span (monotone in the direction of integration)
        for n in range(N):
            if (tdir*(tspan[n+1] - tspan[n]) < 0):
                raise ValueError("AdaptARK::Evolve illegal tspan")

        # use the magnitude of any user-supplied step size, limited by the explicit
        # stability limit, signed in direction tdir
        if (self.h != 0.0):
            self.h = tdir*min(abs(self.h), self.hE)

        # initialize error weight vector, and check for legal tolerances
        self.w = self.error_weight(y, self.w)

        # estimate initial step size if not provided by user
        if (self.h == 0.0):

            # get ||y'(t0)||
            fn = self.fE(t, y, *args) + self.fI(t, y, *args)

            # estimate initial h value via linearization, safety factor, and explicit stability limit
            self.error_norm = max(np.linalg.norm(fn*self.w, np.inf), 1.e-8)
            self.h = tdir*min(max(self.hmin, self.safety / self.error_norm), self.hE)

        # iterate over output times
        for iout in range(1,N+1):

            # loop over internal steps to reach desired output time
            while (tdir*(tspan[iout]-t) > np.sqrt(np.finfo(float).eps*abs(tspan[iout]))):

                # enforce maxit -- if we've exceeded attempts, return with failure
                if (self.steps + self.fails > self.maxit):
                    print("AdaptARK: reached maximum iterations, returning with failure")
                    return Y, False

                # bound internal time step to not exceed the explicit stability limit or next output time
                self.h = tdir*min(abs(self.h), self.hE, abs(tspan[iout]-t))

                # reset temporary solution to current solution, and take ARK step
                self.yt = y.copy()
                t, self.yt, success = self.step(t, self.yt, args)
                if (not success):
                    print("AdaptARK::Evolve error in time step at t =", t)
                    return Y, False

                # estimate step size growth/reduction factor based on error estimate
                eta = self.safety * self.error_norm**(-1.0/(self.minpq+1))  # step size growth factor
                eta = min(eta, self.growth)                             # limit maximum growth

                # store step size in history if requested
                if (self.save_step_hist):
                    self.step_hist['t'].append(t)
                    self.step_hist['h'].append(self.h)
                    self.step_hist['err'].append(self.error_norm)

                # check error
                if (self.error_norm < self.ONEPSM):  # successful step

                    # update current time, solution, error weights, work counter, and upcoming stepsize
                    t += self.h
                    y = self.yt.copy()
                    self.w = self.error_weight(y, self.w)
                    self.steps += 1
                    self.h = tdir*min(abs(self.h) * eta, self.hE)

                else:                                 # failed step
                    self.fails += 1

                    # adjust step size, enforcing minimum and returning with failure if needed
                    if (abs(self.h) > self.hmin):                         # failure, but reduction possible
                        self.h = tdir*max(abs(self.h) * eta, self.hmin)
                    else:                                                 # failed with no reduction possible
                        print("AdaptARK: error test failed at h=hmin, returning with failure")
                        return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with successful solution
        return Y, True

    def set_rtol(self, rtol=1e-3):
        """ Resets the relative tolerance """
        self.rtol = rtol

    def set_atol(self, atol=1e-14):
        """ Resets the scalar- or vector-valued absolute tolerance """
        self.atol = np.ones(self.atol.size)*atol

    def set_maxit(self, maxit=1e6):
        """ Resets the maximum allowed iterations """
        self.maxit = maxit

    def set_bias(self, bias=1.0):
        """ Resets the error bias factor """
        self.bias = bias

    def set_growth(self, growth=50.0):
        """ Resets the maximum stepsize growth factor """
        self.growth = growth

    def set_safety(self, safety=0.85):
        """ Resets the stepsize safety factor """
        self.safety = safety

    def set_hmin(self, hmin=10*np.finfo(float).eps):
        """ Resets the minimum step size """
        self.hmin = hmin

    def set_hE(self, hE=np.inf):
        """ Resets the explicit stability step-size limit """
        self.hE = hE

    def update_rhs(self, fE, fI):
        """ Updates the RHS functions (cannot change vector dimensions) """
        self.fE = fE
        self.fI = fI

    def get_error_weight(self):
        """ Returns the current error weight vector """
        return self.w

    def get_error_vector(self):
        """ Returns the current error vector """
        return self.yerr

    def get_error_norm(self):
        """ Returns the scaled error norm """
        return self.error_norm

    def get_num_error_failures(self):
        """ Returns the total number of error test failures """
        return self.fails

    def get_num_steps(self):
        """ Returns the accumulated number of steps """
        return self.steps

    def get_num_solves(self):
        """ Returns the accumulated number of implicit solves """
        return self.nsol

    def get_current_step(self):
        """ Returns the current internal step size (signed, negative when integrating backward) """
        return self.h

    def get_step_history(self):
        """ Returns the current step size history (step sizes h are signed, negative when integrating backward) """
        return self.step_hist

    def reset(self):
        """ Resets the accumulated number of steps """
        self.fails = 0
        self.error_norm = 0.0
        self.nsol = 0
        self.steps = 0
        self.step_hist = {'t': [], 'h': [], 'err': []}


# embedded ARK Butcher table pair routines
sys.path.append('../04_explicit_one_step')
sys.path.append('../05_implicit_one_step')
from AdaptERK import (Ascher222ERK, SSP32ERK, ARK232ERK, SSP2332LspumERK,
                      GiraldoARK2ERK, ARK324L2SAERK, SSP43ERK, SSP93ERK,
                      ARK436L2SAERK, ARK437L2SAERK, ARK548L2SAERK,
                      ARK548L2SAbERK)
from AdaptDIRK import (Ascher222SDIRK, SSP32DIRK, ARK232SDIRK,
                       SSP2332LspumSDIRK, GiraldoARK2ESDIRK, ARK324L2SAESDIRK,
                       SSP43ESDIRK, ARK436L2SAESDIRK,
                       ARK437L2SAESDIRK, ARK548L2SAESDIRK, ARK548L2SAbESDIRK)

def ARS222():
    """
    Usage: BE, BI = ARS222()

    Utility routine to return the embedded ARK Butcher table pair corresponding
    to the ARS(2,2,2) method.

    Outputs: BE holds the explicit embedded Butcher table
             BI holds the implicit embedded Butcher table
    """
    BE = Ascher222ERK()
    BI = Ascher222SDIRK()
    return BE, BI

def SSP32():
    """
    Usage: BE, BI = SSP32()

    Utility routine to return the embedded ARK Butcher table pair corresponding
    to the SSP(3,2) method.

    Outputs: BE holds the explicit embedded Butcher table
             BI holds the implicit embedded Butcher table
    """
    BE = SSP32ERK()
    BI = SSP32DIRK()
    return BE, BI

def ARK232():
    """
    Usage: BE, BI = ARK232()

    Utility routine to return the embedded ARK Butcher table pair corresponding
    to the ARK(2,3,2) method.

    Outputs: BE holds the explicit embedded Butcher table
             BI holds the implicit embedded Butcher table
    """
    BE = ARK232ERK()
    BI = ARK232SDIRK()
    return BE, BI

def SSP2332Lspum():
    """
    Usage: BE, BI = SSP2332Lspum()

    Utility routine to return the embedded ARK Butcher table pair corresponding
    to the SSP2(3,3,2)-lspum method.

    Outputs: BE holds the explicit embedded Butcher table
             BI holds the implicit embedded Butcher table
    """
    BE = SSP2332LspumERK()
    BI = SSP2332LspumSDIRK()
    return BE, BI

def GiraldoARK2():
    """
    Usage: BE, BI = GiraldoARK2()

    Utility routine to return the embedded ARK Butcher table pair corresponding
    to Giraldo's ARK2 method.

    Outputs: BE holds the explicit embedded Butcher table
             BI holds the implicit embedded Butcher table
    """
    BE = GiraldoARK2ERK()
    BI = GiraldoARK2ESDIRK()
    return BE, BI

def ARK324L2SA():
    """
    Usage: BE, BI = ARK324L2SA()

    Utility routine to return the embedded ARK Butcher table pair corresponding
    to Kennedy & Carpenter's ARK3(2)4L[2]SA method.

    Outputs: BE holds the explicit embedded Butcher table
             BI holds the implicit embedded Butcher table
    """
    BE = ARK324L2SAERK()
    BI = ARK324L2SAESDIRK()
    return BE, BI

def SSP43():
    """
    Usage: BE, BI = SSP43()

    Utility routine to return the embedded ARK Butcher table pair corresponding
    to the SSP(4,3) method.

    Outputs: BE holds the explicit embedded Butcher table
             BI holds the implicit embedded Butcher table
    """
    BE = SSP43ERK()
    BI = SSP43ESDIRK()
    return BE, BI

def ARK436L2SA():
    """
    Usage: BE, BI = ARK436L2SA()

    Utility routine to return the embedded ARK Butcher table pair corresponding
    to Kennedy & Carpenter's ARK4(3)6L[2]SA method.

    Outputs: BE holds the explicit embedded Butcher table
             BI holds the implicit embedded Butcher table
    """
    BE = ARK436L2SAERK()
    BI = ARK436L2SAESDIRK()
    return BE, BI

def ARK437L2SA():
    """
    Usage: BE, BI = ARK437L2SA()

    Utility routine to return the embedded ARK Butcher table pair corresponding
    to Kennedy & Carpenter's ARK4(3)7L[2]SA method.

    Outputs: BE holds the explicit embedded Butcher table
             BI holds the implicit embedded Butcher table
    """
    BE = ARK437L2SAERK()
    BI = ARK437L2SAESDIRK()
    return BE, BI

def ARK548L2SA():
    """
    Usage: BE, BI = ARK548L2SA()

    Utility routine to return the embedded ARK Butcher table pair corresponding
    to Kennedy & Carpenter's ARK5(4)8L[2]SA method.

    Outputs: BE holds the explicit embedded Butcher table
             BI holds the implicit embedded Butcher table
    """
    BE = ARK548L2SAERK()
    BI = ARK548L2SAESDIRK()
    return BE, BI

def ARK548L2SAb():
    """
    Usage: BE, BI = ARK548L2SAb()

    Utility routine to return the embedded ARK Butcher table pair corresponding
    to Kennedy & Carpenter's ARK5(4)8L[2]SAb method.

    Outputs: BE holds the explicit embedded Butcher table
             BI holds the implicit embedded Butcher table
    """
    BE = ARK548L2SAbERK()
    BI = ARK548L2SAbESDIRK()
    return BE, BI

# end of file
