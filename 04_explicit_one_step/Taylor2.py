# Taylor2.py
#
# Fixed-stepsize second-order Taylor stepper class implementation file.
#
# Class to perform fixed-stepsize time evolution of the IVP
#      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
# using an explicit second-order Taylor time stepping method.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np

class Taylor2:
    """
    Fixed stepsize second-order Taylor method class

    The three required arguments when constructing a Taylor2 object are
    functions for the IVP right-hand side and its first partial derivatives:
        f = ODE RHS function with calling syntax f(t,y).
        f_t = t-derivative of ODE RHS function with calling syntax f_t(t,y).
        f_y = y-derivative of ODE RHS function with calling syntax f_y(t,y).
        h = (optional) input with stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    """
    def __init__(self, f, f_t, f_y, h=0.0):
        # required inputs
        self.f = f
        self.f_t = f_t
        self.f_y = f_y

        # optional inputs
        self.h = h

        # internal data
        self.steps = 0
        self.nrhs = 0

    def Taylor2_step(self, t, y, args=()):
        """
        Usage: t, y, success = Taylor2_step(t, y, args)

        Utility routine to take a single second-order Taylor method step,
        where the inputs (t,y) are overwritten by the updated versions.
        args is used for optional parameters of the RHS and its derivatives.
        If success==True then the step succeeded; otherwise it failed.
        """

        # evaluate RHS function and its derivatives
        self.fn = self.f(t, y, *args)
        self.ft = self.f_t(t, y, *args)
        self.fy = self.f_y(t, y, *args)
        self.nrhs += 3

        # update time step solution and tcur
        y += self.h * (self.fn + 0.5*self.h*(self.ft + self.fy@self.fn))
        t += self.h
        self.steps += 1
        return t, y, True

    def reset(self):
        """ Resets the accumulated number of steps """
        self.steps = 0
        self.nrhs = 0

    def get_num_steps(self):
        """ Returns the accumulated number of steps """
        return self.steps

    def get_num_rhs(self):
        """ Returns the accumulated number of RHS evaluations """
        return self.nrhs

    def Evolve(self, tspan, y0, h=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, h)

        The fixed-step second-order Taylor method evolution routine

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
            raise ValueError("ERROR: Taylor2::Evolve called without specifying a nonzero step size")

        # verify that tspan values are separated by multiples of h
        for n in range(tspan.size-1):
            hn = tspan[n+1]-tspan[n]
            if (abs(round(hn/self.h) - (hn/self.h)) > 100*np.sqrt(np.finfo(h).eps)*abs(self.h)):
                raise ValueError("input values in tspan (%e,%e) are not separated by a multiple of h = %e" % (tspan[n],tspan[n+1],h))

        # initialize output, and set first entry corresponding to initial condition
        y = y0.copy()
        Y = np.zeros((tspan.size, y0.size))
        Y[0,:] = y

        # initialize internal solution-vector-sized data
        self.fn = y0.copy()
        self.ft = y0.copy()
        self.fy = np.zeros((y0.size, y0.size), dtype=float)

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required
            N = int(round((tspan[iout]-tspan[iout-1])/self.h))

            # reset "current" t that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform explicit Runge--Kutta update
                t, y, success = self.Taylor2_step(t, y, args)
                if (not success):
                    print("erk error in time step at t =", t)
                    return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with "success" flag
        return Y, True
