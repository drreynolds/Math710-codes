# ARK.py
#
# Fixed-stepsize implicit-explicit additive Runge--Kutta stepper class
# implementation file.
#
# Also contains functions to return specific ARK Butcher table pairs.  The
# individual explicit and implicit tables are stored in ERK.py, DIRK.py,
# AdaptERK.py and AdaptDIRK.py; the routines at the end of this file assemble
# them into compatible pairs.
#
# Class to perform fixed-stepsize time evolution of the IVP
#      y' = fE(t,y) + fI(t,y),  t in [t0, Tf],  y(t0) = y0
# using an implicit-explicit additive Runge--Kutta (ARK) time stepping
# method.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from utilities.substeps import substeps
import sys
sys.path.append('..')
from shared.ImplicitSolver import *

class ARK:
    """
    Fixed stepsize implicit-explicit additive Runge--Kutta class

    The five required arguments when constructing an ARK object are
    functions for the explicit and implicit portions of the IVP right-hand
    side, an implicit solver to use, and explicit and implicit Butcher tables:
        fE = explicit ODE RHS function with calling syntax fE(t,y).
        fI = implicit ODE RHS function with calling syntax fI(t,y).
        sol = algebraic solver object to use [ImplicitSolver]
        BE = explicit Runge--Kutta Butcher table.
        BI = diagonally-implicit Runge--Kutta Butcher table.
        h = (optional) input with requested stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    """
    def __init__(self, fE, fI, sol, BE, BI, h=0.0):
        # required inputs
        self.fE = fE
        self.fI = fI
        self.sol = sol
        self.AE = BE['A']
        self.bE = BE['b']
        self.cE = BE['c']
        self.AI = BI['A']
        self.bI = BI['b']
        self.cI = BI['c']

        # optional inputs
        self.h = h

        # internal data
        self.steps = 0
        self.nsol = 0
        self.s = self.cE.size

        # check for legal tables (including matching numbers of stages)
        if ((np.size(self.cI,0) != self.s) or (np.size(self.bE,0) != self.s) or
            (np.size(self.bI,0) != self.s) or (np.size(self.AE,0) != self.s) or
            (np.size(self.AE,1) != self.s) or (np.size(self.AI,0) != self.s) or
            (np.size(self.AI,1) != self.s) or
            (np.linalg.norm(self.AE - np.tril(self.AE,-1), np.inf) > 1e-14) or
            (np.linalg.norm(self.AI - np.tril(self.AI,0), np.inf) > 1e-14)):
            raise ValueError("ARK ERROR: incompatible Butcher tables supplied")

    def ark_step(self, t, y, h, args=()):
        """
        Usage: t, y, success = ark_step(t, y, h, args)

        Utility routine to take a single implicit-explicit ARK time step of size h,
        where the inputs (t,y) are overwritten by the updated versions.
        args is used for optional parameters of the RHS functions.
        If success==True then the step succeeded; otherwise it failed.
        """

        # loop over stages, computing RHS vectors
        for i in range(self.s):

            # construct "data" for this stage solve
            self.data = np.copy(y)
            for j in range(i):
                self.data += h * (self.AE[i,j] * self.kE[j,:]
                                  + self.AI[i,j] * self.kI[j,:])

            # solve the implicit stage (or copy the data for an explicit stage)
            tstageI = t + h*self.cI[i]
            if (abs(self.AI[i,i]) > 1e-14):
                F = lambda zcur: zcur - self.data - h * self.AI[i,i] * self.fI(tstageI, zcur, *args)
                self.sol.setup_linear_solver(tstageI, -h * self.AI[i,i], args)
                self.z, iters, success = self.sol.solve(F, y)
                self.nsol += 1
                if (not success):
                    return t, y, False
            else:
                self.z = self.data.copy()

            # store both RHS vectors at this stage
            self.kE[i,:] = self.fE(t + h*self.cE[i], self.z, *args)
            self.kI[i,:] = self.fI(tstageI, self.z, *args)

        # update time step solution
        for i in range(self.s):
            y += h * (self.bE[i] * self.kE[i,:] + self.bI[i] * self.kI[i,:])
        t += h
        self.steps += 1
        return t, y, True

    def update_rhs(self, fE, fI):
        """ Updates the RHS functions (cannot change vector dimensions) """
        self.fE = fE
        self.fI = fI

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

        The fixed-step ARK evolution routine

        Inputs:  tspan holds the current time interval, [t0, tf], including any
                     intermediate times when the solution is desired, i.e.
                     [t0, t1, ..., tf]
                 y holds the initial condition, y(t0)
                 h optionally holds the requested step size (if it is not
                     provided then the stored value will be used)
                 args holds optional equation parameters used when evaluating
                     the RHS functions.
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
            raise ValueError("ERROR: ARK::Evolve called without specifying a nonzero step size")

        # initialize output, and set first entry corresponding to initial condition
        y = y0.copy()
        Y = np.zeros((tspan.size,y0.size))
        Y[0,:] = y

        # initialize internal solution-vector-sized data
        self.kE = np.zeros((self.s, y0.size), dtype=float)
        self.kI = np.zeros((self.s, y0.size), dtype=float)
        self.z = y0.copy()
        self.data = y0.copy()

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required, and the actual step size to use
            N, h = substeps(tspan[iout]-tspan[iout-1], self.h)

            # reset "current" t that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform implicit-explicit additive Runge--Kutta update
                t, y, success = self.ark_step(t, y, h, args)
                if (not success):
                    print("ARK::Evolve error in time step at t =", t)
                    return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with "success" flag
        return Y, True

# ARK Butcher table pair routines
sys.path.append('../04_explicit_one_step')
sys.path.append('../05_implicit_one_step')
from ERK import (Heun, ERK4, Ascher111ERK, Ascher122ERK, ARKCouplingERK3,
                 Ascher232ERK, Ascher233ERK, Ascher343ERK, Ascher443ERK,
                 SSP222ERK, SSP2332Lpm1ERK, SSP2332Lpm2ERK, SSP2332LpumERK,
                 SSP2332aERK, DBM53ERK, Cooper4ERK, SSP3433ERK, Cooper6ERK)
from DIRK import (Ascher111SDIRK, Ascher122SDIRK, ARKCouplingESDIRK3,
                  Ascher232SDIRK, Ascher233SDIRK, Ascher343SDIRK,
                  Ascher443PaddedSDIRK, SSP222SDIRK, SSP2332Lpm1SDIRK,
                  SSP2332Lpm2SDIRK, SSP2332LpumSDIRK, SSP2332aDIRK,
                  DBM53ESDIRK, Cooper4ESDIRK, SSP3433SDIRK, Cooper6ESDIRK)
from AdaptERK import Ascher222ERK, ARK324L2SAERK, ARK436L2SAERK
from AdaptDIRK import Ascher222SDIRK, ARK324L2SAESDIRK, ARK436L2SAESDIRK

def ARS111():
    """
    Usage: BE, BI = ARS111()

    Utility routine to return the ARK Butcher table pair corresponding
    to the ARS(1,1,1) forward-backward Euler method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = Ascher111ERK()
    BI = Ascher111SDIRK()
    return BE, BI

def ARS122():
    """
    Usage: BE, BI = ARS122()

    Utility routine to return the ARK Butcher table pair corresponding
    to the ARS(1,2,2) implicit-explicit midpoint method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = Ascher122ERK()
    BI = Ascher122SDIRK()
    return BE, BI

def ARS222():
    """
    Usage: BE, BI = ARS222()

    Utility routine to return the ARK Butcher table pair corresponding
    to the ARS(2,2,2) method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = Ascher222ERK()
    BI = Ascher222SDIRK()
    return BE, BI

def ARS232():
    """
    Usage: BE, BI = ARS232()

    Utility routine to return the ARK Butcher table pair corresponding
    to the ARS(2,3,2) method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = Ascher232ERK()
    BI = Ascher232SDIRK()
    return BE, BI

def ARS233():
    """
    Usage: BE, BI = ARS233()

    Utility routine to return the ARK Butcher table pair corresponding
    to the ARS(2,3,3) method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = Ascher233ERK()
    BI = Ascher233SDIRK()
    return BE, BI

def ARS343():
    """
    Usage: BE, BI = ARS343()

    Utility routine to return the ARK Butcher table pair corresponding
    to the ARS(3,4,3) method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = Ascher343ERK()
    BI = Ascher343SDIRK()
    return BE, BI

def ARS443():
    """
    Usage: BE, BI = ARS443()

    Utility routine to return the ARK Butcher table pair corresponding
    to the ARS(4,4,3) method, with padded SDIRK table.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = Ascher443ERK()
    BI = Ascher443PaddedSDIRK()
    return BE, BI

def SSP222():
    """
    Usage: BE, BI = SSP222()

    Utility routine to return the ARK Butcher table pair corresponding
    to the SSP2(2,2,2) method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = SSP222ERK()
    BI = SSP222SDIRK()
    return BE, BI

def SSP2332Lpm1():
    """
    Usage: BE, BI = SSP2332Lpm1()

    Utility routine to return the ARK Butcher table pair corresponding
    to the SSP2(3,3,2)-lpm1 method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = SSP2332Lpm1ERK()
    BI = SSP2332Lpm1SDIRK()
    return BE, BI

def SSP2332Lpm2():
    """
    Usage: BE, BI = SSP2332Lpm2()

    Utility routine to return the ARK Butcher table pair corresponding
    to the SSP2(3,3,2)-lpm2 method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = SSP2332Lpm2ERK()
    BI = SSP2332Lpm2SDIRK()
    return BE, BI

def SSP2332Lpum():
    """
    Usage: BE, BI = SSP2332Lpum()

    Utility routine to return the ARK Butcher table pair corresponding
    to the SSP2(3,3,2)-lpum method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = SSP2332LpumERK()
    BI = SSP2332LpumSDIRK()
    return BE, BI

def SSP2332a():
    """
    Usage: BE, BI = SSP2332a()

    Utility routine to return the ARK Butcher table pair corresponding
    to the SSP2(3,3,2)-a method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = SSP2332aERK()
    BI = SSP2332aDIRK()
    return BE, BI

def DBM53():
    """
    Usage: BE, BI = DBM53()

    Utility routine to return the ARK Butcher table pair corresponding
    to the DBM-5-3 method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = DBM53ERK()
    BI = DBM53ESDIRK()
    return BE, BI

def Cooper4():
    """
    Usage: BE, BI = Cooper4()

    Utility routine to return the ARK Butcher table pair corresponding
    to the Cooper4 method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = Cooper4ERK()
    BI = Cooper4ESDIRK()
    return BE, BI

def SSP3433():
    """
    Usage: BE, BI = SSP3433()

    Utility routine to return the ARK Butcher table pair corresponding
    to the SSP3(4,3,3) method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = SSP3433ERK()
    BI = SSP3433SDIRK()
    return BE, BI

def Cooper6():
    """
    Usage: BE, BI = Cooper6()

    Utility routine to return the ARK Butcher table pair corresponding
    to the Cooper6 method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = Cooper6ERK()
    BI = Cooper6ESDIRK()
    return BE, BI

def ARK324L2SA():
    """
    Usage: BE, BI = ARK324L2SA()

    Utility routine to return the ARK Butcher table pair corresponding
    to Kennedy & Carpenter's ARK3(2)4L[2]SA method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = ARK324L2SAERK()
    BI = ARK324L2SAESDIRK()
    return BE, BI

def ARK436L2SA():
    """
    Usage: BE, BI = ARK436L2SA()

    Utility routine to return the ARK Butcher table pair corresponding
    to Kennedy & Carpenter's ARK4(3)6L[2]SA method.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = ARK436L2SAERK()
    BI = ARK436L2SAESDIRK()
    return BE, BI

def HeunImplicitMidpoint():
    """
    Usage: BE, BI = HeunImplicitMidpoint()

    Utility routine to return the ARK Butcher table pair corresponding
    to the first-order pairing of Heun's method with the padded
    implicit midpoint method, from the ARK coupling example.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = Heun()
    BI = Ascher122SDIRK()
    return BE, BI

def RK4ESDIRK3():
    """
    Usage: BE, BI = RK4ESDIRK3()

    Utility routine to return the ARK Butcher table pair corresponding
    to the second-order pairing of the classical RK4 method with
    a third-order ESDIRK method, from the ARK coupling example.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = ERK4()
    BI = ARKCouplingESDIRK3()
    return BE, BI

def ERK3ESDIRK3():
    """
    Usage: BE, BI = ERK3ESDIRK3()

    Utility routine to return the ARK Butcher table pair corresponding
    to the third-order pairing of a third-order ERK method with
    a third-order ESDIRK method, from the ARK coupling example.

    Outputs: BE holds the explicit Butcher table
             BI holds the implicit Butcher table
    """
    BE = ARKCouplingERK3()
    BI = ARKCouplingESDIRK3()
    return BE, BI

# end of file
