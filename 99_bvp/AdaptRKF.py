# AdaptRKF.py
#
# Adaptive Runge--Kutta--Fehlberg solver class implementation file.
#
# Class to perform adaptive time evolution of the IVP
#      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
# using the Runge--Kutta--Fehlberg time stepping method.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np

class AdaptRKF:
    """
    Adaptive Runge--Kutta--Fehlberg class

    The two required arguments when constructing an AdaptRKF object
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
        self.p = 4
        self.fails = 0
        self.steps = 0
        self.nrhs = 0
        self.error_norm = 0.0
        self.h = 0.0
        self.z = np.zeros(y.size)
        self.yt = np.zeros(y.size)
        self.k = np.zeros((6, y.size))
        self.A = np.array([[0, 0, 0, 0, 0, 0],
                           [1/4, 0, 0, 0, 0, 0],
                           [3/32, 9/32, 0, 0, 0, 0],
                           [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
                           [439/216, -8, 3680/513, -845/4104, 0, 0],
                           [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]],
                          dtype=float)
        self.b = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],
                          dtype=float)
        self.c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2], dtype=float)
        self.d = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0], dtype=float)
        self.s = 6

    def error_weight(self, y, w):
        """
        Error weight vector utility routine
        """
        for i in range(y.size):
            w[i] = self.bias / (self.atol[i] + self.rtol * np.abs(y[i]))
        return w

    def RKF_step(self, t, y, args=()):
        """
        Usage: t, y, success = RKF_step(t, y, args)

        Utility routine to take a single explicit RK time step,
        where the inputs (t,y) are overwritten by the updated versions.
        args holds optional parameters used when evaluating the RHS.
        If success==True then the step succeeded; otherwise it failed.
        """

        # loop over stages, computing RHS vectors
        self.k[0,:] = self.f(t, y, *args)
        self.nrhs += 1
        for i in range(1,self.s):
            self.z = np.copy(y)
            for j in range(i):
                self.z += self.h * self.A[i,j] * self.k[j,:]
            self.k[i,:] = self.f(t + self.c[i] * self.h, self.z, *args)
            self.nrhs += 1

        # update time step solution
        for i in range(self.s):
            y += self.h * self.b[i] * self.k[i,:]

        # compute error estimate (and norm), and return
        self.yerr *= 0.0
        for i in range(self.s):
            self.yerr += self.h * (self.b[i] - self.d[i])* self.k[i,:]
        self.error_norm = max(np.linalg.norm(self.yerr*self.w, np.inf), 1.e-8)
        return t, y, True

    def Evolve(self, tspan, y0, h=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, h, args)

        The adaptive RKF time step evolution routine

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
                raise ValueError("AdaptRKF::Evolve illegal tspan")

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
                    print("AdaptRKF: reached maximum iterations, returning with failure")
                    return Y, False

                # bound internal time step to not exceed next output time
                self.h = min(self.h, tspan[iout]-t)

                # reset temporary solution to current solution, and take RKF step
                self.yt = y.copy()
                t, yt, success = self.RKF_step(t, self.yt, args)

                # estimate step size growth/reduction factor based on error estimate
                eta = self.safety * self.error_norm**(-1.0/(self.p+1))  # step size growth factor
                eta = min(eta, self.growth)                             # limit maximum growth

                # check error
                if (self.error_norm < self.ONEPSM):  # successful step

                    # update current time, solution, error weights, work counter, and upcoming stepsize
                    t += self.h
                    y = self.yt.copy()
                    self.w = self.error_weight(y, self.w)
                    self.steps += 1
                    self.h *= eta

                else:                                 # failed step
                    self.fails += 1

                    # adjust step size, enforcing minimum and returning with failure if needed
                    if (self.h > self.hmin):                              # failure, but reduction possible
                        self.h = max(self.h * eta, self.hmin)
                    else:                                                 # failed with no reduction possible
                        print("AdaptRKF: error test failed at h=hmin, returning with failure")
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

    def get_num_rhs(self):
        """ Returns the total number of rhs calls """
        return self.nrhs

    def get_current_step(self):
        """ Returns the current internal step size """
        return self.h

    def reset(self):
        """ Resets the solver statistics """
        self.fails = 0
        self.steps = 0
