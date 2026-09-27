#!/usr/bin/env python
#
# This file defines the RHS and Jacobian functions associated with the
# viscous Burgers equation,
#
#    u_t + (u^2/2)_x = nu*u_xx,  x in [-1,1], t in [0,1],
#    u(t,-1) = g(-1,t),  u(t,1) = g(1,t),
#    u(0,x) = g(x,0),
#
# that has the traveling-wave analytical solution
#
#    g(x,t) = (1 - tanh((x - t/2)/(4*nu)))/2,
#
# a viscous shock connecting u=1 to u=0 that moves to the right with speed 1/2.
# We discretize in space using centered, second-order differences on a uniform
# mesh, with the Dirichlet boundary values taken from g.  So that computed
# errors measure only the temporal error, we add a forcing function Phi(t) to
# the spatially-discretized problem,
#
#    u' = -D1 (u^2/2) + nu*D2 u + Phi(t),
#
# chosen so that the semi-discrete solution is exactly u_j(t) = g(x_j,t) at
# every interior grid point.  For this problem,
#
#    Phi(t) = g_t + D1 (g^2/2) - nu*D2 g,
#
# where g and g_t = sech^2((x - t/2)/(4*nu))/(16*nu) are evaluated on the
# spatial grid, and where D1 and D2 include the boundary values of g.  Since g
# solves the PDE, Phi(t) is just the spatial truncation error.
#
# Note that the Jacobian of the advection term is -D1 diag(u).
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import scipy.sparse as sp

# problem parameters
t0 = 0.0
tf = 1.0
xl = -1.0
xr = 1.0
Nx = 401
nu = 5.e-2
dx = (xr-xl)/(Nx-1)
xgrid = np.linspace(xl+dx, xr-dx, Nx-2)

# first- and second-derivative matrices for the interior unknowns (in sparse format)
D1 = sp.diags_array((-np.ones(Nx-3)/(2*dx), np.ones(Nx-3)/(2*dx)),
                    offsets=(-1,1), format='csc')
D2 = sp.diags_array((np.ones(Nx-3)/(dx**2), -2*np.ones(Nx-2)/(dx**2),
                     np.ones(Nx-3)/(dx**2)), offsets=(-1,0,1), format='csc')

# constant Jacobian of the implicit portion
JI_matrix = nu*D2

def utrue(x,t):
    """ True solution to the spatially-discretized problem. """
    return 0.5*(1.0 - np.tanh((x - 0.5*t)/(4*nu)))

def utrue_t(x,t):
    """ Time derivative of the true solution. """
    return 1.0/(16*nu*np.cosh((x - 0.5*t)/(4*nu))**2)

def advection(t,u):
    """ Centered approximation of -(u^2/2)_x, including the boundary values. """
    F = 0.5*u**2
    val = -(D1 @ F)
    val[0] += 0.5*utrue(xl,t)**2/(2*dx)
    val[-1] -= 0.5*utrue(xr,t)**2/(2*dx)
    return val

def diffusion(t,u):
    """ Centered approximation of nu*u_xx, including the boundary values. """
    val = JI_matrix @ u
    val[0] += nu*utrue(xl,t)/(dx**2)
    val[-1] += nu*utrue(xr,t)/(dx**2)
    return val

def Phi(t):
    """ Forcing function for the spatially-discretized problem. """
    g = utrue(xgrid,t)
    return utrue_t(xgrid,t) - advection(t,g) - diffusion(t,g)

def fE(t,u):
    """ Explicit advection portion of the right-hand side. """
    return advection(t,u)

def fI(t,u):
    """ Implicit diffusion and forcing portion of the right-hand side. """
    return diffusion(t,u) + Phi(t)

def JI(t,u):
    """ Jacobian of the implicit diffusion and forcing portion of the right-hand side. """
    return JI_matrix

def f(t,u):
    """ Full right-hand side function for the IVP. """
    return fE(t,u) + fI(t,u)

def J(t,u):
    """ Jacobian of the full right-hand side function. """
    return (-(D1 @ sp.diags_array(u)) + JI_matrix).tocsc()

def u0():
    """ Initial condition. """
    return utrue(xgrid,t0)

def explicit_stability_limit(cfl=2.4):
    """
    Step-size limit for the explicit advection term, h_E = cfl*dx/max|u|, where
    max|u| = 1 for this solution.  The default cfl value is slightly below the
    stability limit (approximately 2.58) of the explicit ARK3(2)4L[2]SA table on
    the imaginary axis.
    """
    return cfl*dx/1.0

# end of file
