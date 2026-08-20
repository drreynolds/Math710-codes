# AdaptEuler.py
#
# Adaptive forward Euler solver class implementation file.
#
# Class to perform adaptive time evolution of the IVP
#      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
# using the forward Euler (explicit Euler) time stepping method.
#
# D.R. Reynolds
# Math 6321 @ SMU
# Fall 2023
import numpy as np

class AdaptEuler:
    """
    Adaptive forward Euler class

    The two required arguments when constructing an AdaptEuler object
    are a function for the IVP right-hand side, and a template vector
    with the same shape and type as the IVP solution vector:
      f = ODE RHS function with calling syntax f(t,y).
      y = numpy array with m entries.

    Other optional inputs focus on specific adaptivity options:
      rtol    = relative solution tolerance (float, >= 1e-12)
      atol    = absolute solution tolerance (float or numpy array with m entries, all >=0)
      maxit   = maximum allowed number of internal steps
      bias    = error bias factor
      growth  = maximum stepsize growth factor
      safety  = step size safety factor
    """
    def __init__(self, f, y, rtol=1e-3, atol=1e-14, maxit=1e6, bias=2.0, growth=50.0, safety=0.95, hmin=10*np.finfo(float).eps):
        # required inputs
        self.f = f
        # optional inputs
        self.rtol = rtol
        self.atol = np.ones(y.size)*atol
        self.maxit = maxit
        self.bias = bias
        self.growth = growth
        self.safety = safety
        self.hmin = hmin
        # internal data
        self.w = np.ones(y.size)
        self.yerr = np.zeros(y.size)
        self.ONEMSM = 1.0 - np.sqrt(np.finfo(float).eps)
        self.ONEPSM = 1.0 + np.sqrt(np.finfo(float).eps)
        self.p = 1
        self.fails = 0
        self.steps = 0
        self.error_norm = 0.0
        self.h = 0.0

    def error_weight(self, y, w):
        """
        Error weight vector utility routine
        """
        for i in range(y.size):
            w[i] = self.bias / (self.atol[i] + self.rtol * np.abs(y[i]))
        return w

    def Evolve(self, tspan, y0, h=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, h, args)

        The adaptive forward Euler time step evolution routine

        Inputs:  tspan holds the current time interval, [t0, tf], including any
                    intermediate times when the solution is desired, i.e.
                     [t0, t1, ..., tf]
                 y holds the initial condition, y(t0)
                 h optionally holds the requested initial step size
                 args holds optional equation parameters used when evaluating
                     the RHS.
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

        # check for legal time span
        for n in range(N):
            if (tspan[n+1] < tspan[n]):
                raise ValueError("AdaptEuler::Evolve illegal tspan")

        # initialize error weight vector, and check for legal tolerances
        self.w = self.error_weight(y, self.w)

        # estimate initial step size if not provided by user
        if (self.h == 0.0):

            # get ||y'(t0)||
            fn = self.f(t, y, *args)

            # estimate initial h value via linearization, safety factor
            self.error_norm = max(np.linalg.norm(fn*self.w, np.inf), 1.e-8)
            self.h = max(self.hmin, self.safety / self.error_norm)

        # iterate over output times
        for iout in range(1,N+1):

            # loop over internal steps to reach desired output time
            while ((tspan[iout]-t) > np.sqrt(np.finfo(float).eps*tspan[iout])):

                # enforce maxit -- if we've exceeded attempts, return with failure
                if (self.steps + self.fails > self.maxit):
                    print("AdaptEuler: reached maximum iterations, returning with failure")
                    return Y, False

                # bound internal time step to not exceed next output time
                self.h = min(self.h, tspan[iout]-t)

                # initialize two solution approximations to current solution
                y1 = y.copy()
                y2 = y.copy()

                # get RHS at this time, perform full/half step updates
                fn = self.f(t, y, *args)
                y1 += self.h*fn
                y2 += (0.5*self.h)*fn

                # get RHS at half-step, perform half step update
                fn = self.f(t+0.5*self.h, y2, *args)
                y2 += (0.5*self.h)*fn

                # compute error estimate
                self.yerr = y2 - y1

                # compute error estimate success factor
                self.error_norm = max(np.linalg.norm(self.yerr*self.w, np.inf), 1.e-8)

                # estimate step size growth/reduction factor based on this error estimate
                eta = self.safety * self.error_norm**(-1.0/(self.p+1))  # step size growth factor
                eta = min(eta, self.growth)                             # limit maximum growth

                # check error
                if (self.error_norm < self.ONEPSM):  # successful step

                    # update current time, solution, error weights, work counter, and upcoming stepsize
                    t += self.h
                    y = 2.0*y2 - y1
                    self.w = self.error_weight(y, self.w)
                    self.steps += 1
                    self.h *= eta

                else:                                 # failed step
                    self.fails += 1

                    # adjust step size, enforcing minimum and returning with failure if needed
                    if (self.h > self.hmin):                              # failure, but reduction possible
                        self.h = max(self.h * eta, self.hmin)
                    else:                                                 # failed with no reduction possible
                        print("AdaptEuler: error test failed at h=hmin, returning with failure")
                        return Y, False

            # store updated solution in output array
            Y[iout,:] = y

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

    def set_bias(self, bias=2.0):
        """ Resets the error bias factor """
        self.bias = bias

    def set_growth(self, growth=50.0):
        """ Resets the maximum stepsize growth factor """
        self.growth = growth

    def set_safety(self, safety=0.95):
        """ Resets the stepsize safety factor """
        self.safety = safety

    def set_hmin(self, hmin=10*np.finfo(float).eps):
        """ Resets the minimum step size """
        self.hmin = hmin

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
        """ Returns the total number of internal time steps """
        return self.steps

    def get_current_step(self):
        """ Returns the current internal step size """
        return self.h

    def reset(self):
        """ Resets the solver statistics """
        self.fails = 0
        self.steps = 0
