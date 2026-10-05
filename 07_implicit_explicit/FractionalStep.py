# FractionalStep.py
#
# Fixed-stepsize fractional-step (operator-splitting) time stepper class
# implementation file.
#
# Class to perform fixed-stepsize time evolution of the M-way split IVP
#      y' = f^{1}(t,y) + f^{2}(t,y) + ... + f^{M}(t,y),  t in [t0, Tf],  y(t0) = y0
# using a fractional-step method
#      Psi_H = Phi^{s}_{a_s H} o ... o Phi^{1}_{a_1 H},
#      Phi^{k}_{a_k H} = phi^{M}_{alpha_k^{M} H} o ... o phi^{1}_{alpha_k^{1} H},
# where each sub-flow phi^{l} is approximated by any object that supports the
# "Evolve" routine for the sub-IVP
#      y' = f^{l}(t,y),  t in [t_k, t_k + alpha_k^{l} H].
#
# Also contains functions to return specific fractional-step coefficients.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from utilities.substeps import substeps

class FractionalStep:
    """
    Fixed stepsize fractional-step time stepper class

    The two required arguments when constructing a FractionalStep object
    are a table of fractional-step coefficients, and a list of solvers for
    the sub-IVPs, one per partition:
        S = fractional-step coefficient table (e.g., S = StrangMarchuk()),
            with the field
            S['alpha'] = M x s array, where alpha[l,k] holds the fraction
                alpha_{k+1}^{l+1} of the step taken by partition l+1 in
                stage k+1 (the same layout as the table of fractional-step
                methods in the lecture notes, with one row per partition).
        Solvers = list of M objects that implement the "Evolve" method, where
                Solvers[l] evolves the sub-IVP y' = f^{l+1}(t,y).  Each solver
                stores its own right-hand side function and, for fixed-step
                solvers, its own step size.
        H = (optional) input with requested stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.

    Within each stage, partition 1 is advanced first.  Sub-steps with
    alpha_k^{l} = 0 are skipped.  A sub-step with alpha_k^{l} < 0 asks its
    solver to evolve backward in time; all of the course solvers (fixed-step
    and adaptive) support this, although a backward sub-step of a dissipative
    piece (e.g., diffusion) is considered inherently unstable.
    """
    def __init__(self, S, Solvers, H=0.0):
        # required inputs
        self.alpha = np.array(S['alpha'], dtype=float)
        self.Solvers = Solvers

        # optional inputs
        self.H = H

        # internal data
        self.steps = 0
        self.M = np.size(self.alpha,0)
        self.s = np.size(self.alpha,1)

        # check for legal inputs
        if (len(self.Solvers) != self.M):
            raise ValueError("FractionalStep ERROR: need one solver per partition")
        for solver in self.Solvers:
            if not hasattr(solver, 'Evolve'):
                raise ValueError("FractionalStep ERROR: each solver must implement the Evolve method")
        if (np.linalg.norm(np.sum(self.alpha,1) - 1.0, np.inf) > 1e-14):
            raise ValueError("FractionalStep ERROR: each partition must advance a total time of H")

    def step(self, t, y, H, args=()):
        """
        Usage: t, y, success = step(t, y, H, args)

        Utility routine to take a single fractional-step time step of size H,
        where the inputs (t,y) are overwritten by the updated versions.
        args is used for optional parameters of the RHS.
        If success==True then the step succeeded; otherwise it failed.
        """

        # tau[l] holds how far partition l+1 has been advanced within this
        # step, so that each sub-IVP starts at that partition's own time
        tau = np.zeros(self.M)

        # loop over stages, and over partitions within each stage
        for k in range(self.s):
            for l in range(self.M):

                # skip sub-steps of zero length
                if (self.alpha[l,k] == 0.0):
                    continue

                # call the partition's solver to evolve its sub-IVP
                tspan = np.array([t + tau[l]*H, t + (tau[l] + self.alpha[l,k])*H])
                ytmp, success = self.Solvers[l].Evolve(tspan, y, args=args)
                if (not success):
                    self.steps += 1
                    return t, y, success
                y = ytmp[1,:]  # extract the solution at the end of the sub-step
                tau[l] += self.alpha[l,k]

        # update current time and step counter, and return
        t += H
        self.steps += 1
        return t, y, True

    def reset(self):
        """ Resets the accumulated number of steps """
        self.steps = 0
        for solver in self.Solvers:
            if hasattr(solver, 'reset'):
                solver.reset()

    def get_num_steps(self):
        """ Returns the accumulated number of fractional steps """
        return self.steps

    def get_num_rhs(self):
        """ Returns the accumulated number of RHS evaluations, over all partitions """
        return sum(solver.get_num_rhs() for solver in self.Solvers if hasattr(solver, 'get_num_rhs'))

    def Evolve(self, tspan, y0, H=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, H, args)

        The fixed-step fractional-step evolution routine

        Inputs:  tspan holds the current time interval, [t0, tf], including any
                     intermediate times when the solution is desired, i.e.
                     [t0, t1, ..., tf]
                 y holds the initial condition, y(t0)
                 H optionally holds the requested step size (if it is not
                     provided then the stored value will be used)
                 args holds optional equation parameters used when evaluating
                     the RHS.
        Outputs: Y holds the computed solution at all tspan values,
                     [y(t0), y(t1), ..., y(tf)]
                 success = True if the solver traversed the interval,
                     false if an integration step failed [bool]
        """

        # set time step for evoluation based on input-vs-stored value
        if (H != 0.0):
            self.H = H

        # raise error if step size was never set
        if (self.H == 0.0):
            raise ValueError("ERROR: FractionalStep::Evolve called without specifying a nonzero step size")

        # initialize output, and set first entry corresponding to initial condition
        y = y0.copy()
        Y = np.zeros((tspan.size,y0.size))
        Y[0,:] = y

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required, and the actual step size to use
            N, H = substeps(tspan[iout]-tspan[iout-1], self.H)

            # reset "current" t that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform fractional step
                t, y, success = self.step(t, y, H, args)
                if (not success):
                    print("FractionalStep error in time step at t =", t)
                    return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with "success" flag
        return Y, True


def LieTrotter(M=2):
    """
    Usage: S = LieTrotter(M)

    Utility routine to return the coefficients of the first-order Lie--Trotter
    splitting for M partitions (default 2), which advances each partition
    once, over the full step, in the order 1, 2, ..., M.

    Reference: lecture notes, table of fractional-step methods.

    Outputs: S['alpha'] holds the fractional-step coefficients
             S['p'] holds the splitting order
    """
    alpha = np.ones((M, 1))
    S = {'alpha': alpha, 'p': 1}
    return S

def LieTrotterAdjoint(M=2):
    """
    Usage: S = LieTrotterAdjoint(M)

    Utility routine to return the coefficients of the adjoint of the
    Lie--Trotter splitting for M partitions (default 2), which advances each
    partition once, over the full step, in the order M, M-1, ..., 1.  Since
    partition 1 is advanced first within each stage, this uses M stages,
    where stage k advances only partition M+1-k.

    Reference: lecture notes, discussion of Lie--Trotter splitting.

    Outputs: S['alpha'] holds the fractional-step coefficients
             S['p'] holds the splitting order
    """
    alpha = np.zeros((M, M))
    for l in range(M):
        alpha[l, M-1-l] = 1.0
    S = {'alpha': alpha, 'p': 1}
    return S

def StrangMarchuk(M=2):
    """
    Usage: S = StrangMarchuk(M)

    Utility routine to return the coefficients of the second-order symmetric
    Strang--Marchuk splitting for M partitions (default 2): a half step of
    partitions 1, ..., M-1, a full step of partition M, and then half steps of
    partitions M-1, ..., 1.  Since partition 1 is advanced first within each
    stage, this uses M stages, where stage 1 advances every partition and
    stage k > 1 advances only partition M+1-k.  For M = 2 these are the
    coefficients in the lecture notes.

    Reference: lecture notes, table of fractional-step methods.

    Outputs: S['alpha'] holds the fractional-step coefficients
             S['p'] holds the splitting order
    """
    alpha = np.zeros((M, M))
    alpha[:, 0] = 0.5
    alpha[M-1, 0] = 1.0
    for l in range(M-1):
        alpha[l, M-1-l] = 0.5
    S = {'alpha': alpha, 'p': 2}
    return S

def OS2(mu):
    """
    Usage: S = OS2(mu)

    Utility routine to return the coefficients of the two-stage,
    second-order, 2-way splitting OS2(2,2)-mu, for a parameter mu != 1.
    All coefficients are positive for 0 < mu < 1/2; mu = 1/2 gives the
    Strang--Marchuk splitting with the roles of the partitions exchanged.

    Reference: Spiteri & Wei, J. Comput. Phys. 476:111900 (2023), Table 1,
               doi:10.1016/j.jcp.2022.111900.

    Outputs: S['alpha'] holds the fractional-step coefficients
             S['p'] holds the splitting order
    """
    if (mu == 1.0):
        raise ValueError("OS2 ERROR: requires mu != 1")
    alpha = np.array((((2*mu-1)/(2*mu-2), -1/(2*mu-2)),
                      (1-mu, mu)), dtype=float)
    S = {'alpha': alpha, 'p': 2}
    return S

def Ruth():
    """
    Usage: S = Ruth()

    Utility routine to return the coefficients of the three-stage,
    third-order, 2-way Ruth splitting.  Note that each partition takes one
    backward (negative) sub-step.

    Reference: Spiteri & Wei, J. Comput. Phys. 476:111900 (2023), Table 3,
               doi:10.1016/j.jcp.2022.111900.

    Outputs: S['alpha'] holds the fractional-step coefficients
             S['p'] holds the splitting order
    """
    alpha = np.array(((7/24, 3/4, -1/24),
                      (2/3, -2/3, 1)), dtype=float)
    S = {'alpha': alpha, 'p': 3}
    return S

def OS3_32():
    """
    Usage: S = OS3_32()

    Utility routine to return the coefficients of the three-stage,
    second-order, 3-way splitting OS3(3,2).  Note that partitions 2 and 3
    each take one backward (negative) sub-step.

    Reference: Spiteri & Wei, J. Comput. Phys. 476:111900 (2023), Table 2,
               doi:10.1016/j.jcp.2022.111900.

    Outputs: S['alpha'] holds the fractional-step coefficients
             S['p'] holds the splitting order
    """
    alpha = np.array(((1/3, 1/3, 1/3),
                      (1, -1/2, 1/2),
                      (1/4, 1, -1/4)), dtype=float)
    S = {'alpha': alpha, 'p': 2}
    return S
