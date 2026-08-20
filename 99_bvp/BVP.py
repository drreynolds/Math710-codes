# BVP.py
#
# Class containing parameters and functions to define the second-order, scalar-valued BVP:
#
#    u'' - 2*lam*u' + lam^2*u = r(t),  0<t<1,
#
#    r(t) = (4*lam^2*exp(lam*(1-t)))/(1+2*exp(lam))
#           + (lam^2-pi^2)*cos(pi*t) + 2*lam*pi*sin(pi*t)
#    u(0) = (1+exp(lam))/(1+2*exp(lam)) + 1
#    u(1) = (1+exp(lam))/(1+2*exp(lam)) - 1
#
# This problem has analytical solution
#
#    u(t) = exp(lam)/(1+2*exp(lam))*(exp(lam*(t-1))
#            + exp(-lam*t)) + cos(pi*t)
#
# and the stiffness may be adjusted using the real-valued
# parameter lam<0
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np

class BVP:
    """
    The one argument when constructing a BVP class is the stiffness parameter, lam.
    """
    def __init__(self, lam):
        # required inputs
        self.lam = lam
        # other parameters
        self.a = 0.0
        self.b = 1.0
        self.ua = (1.0+np.exp(lam))/(1.0+2.0*np.exp(lam)) + 1.0
        self.ub = (1.0+np.exp(lam))/(1.0+2.0*np.exp(lam)) - 1.0

    def r(self, t):
        return ((4*self.lam**2*np.exp(self.lam*(1-t))/(1+2*np.exp(self.lam)))
                + (self.lam**2 - np.pi*np.pi)*np.cos(np.pi*t)
                + 2*self.lam*np.pi*np.sin(np.pi*t))
    def p(self, t):
        return (2*self.lam)
    def q(self, t):
        return (-self.lam**2)
    def utrue(self, t):
        return ( np.exp(self.lam)/(1+2*np.exp(self.lam))*(np.exp(self.lam*(t-1)) + np.exp(-self.lam*t))
                 + np.cos(np.pi*t) )
