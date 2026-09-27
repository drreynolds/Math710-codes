# DIRK.py
#
# Fixed-stepsive diagonally-implicit Runge--Kutta stepper class
# implementation file.
#
# Also contains functions to return specific DIRK Butcher tables.
#
# Class to perform fixed-stepsize time evolution of the IVP
#      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
# using a diagonally-implicit Runge--Kutta (DIRK) time stepping
# method.
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

import numpy as np
import sys
sys.path.append('..')
from shared.ImplicitSolver import *

class DIRK:
    """
    Fixed stepsize diagonally-implicit Runge--Kutta class

    The three required arguments when constructing a DIRK object are a
    function for the IVP right-hand side, an implicit solver to use,
    and a Butcher table:
        f = ODE RHS function with calling syntax f(t,y).
        sol = algebraic solver object to use [ImplicitSolver]
        B = diagonally-implicit Runge--Kutta Butcher table.
        h = (optional) input with requested stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    """
    def __init__(self, f, sol, B, h=0.0):
        # required inputs
        self.f = f
        self.sol = sol
        self.A = B['A']
        self.b = B['b']
        self.c = B['c']

        # optional inputs
        self.h = h

        # internal data
        self.steps = 0
        self.nsol = 0
        self.s = self.c.size

        # check for legal table
        if ((np.size(self.c,0) != self.s) or (np.size(self.A,0) != self.s) or
            (np.size(self.A,1) != self.s) or (np.linalg.norm(self.A - np.tril(self.A,0), np.inf) > 1e-14)):
            raise ValueError("DIRK ERROR: incompatible Butcher table supplied")

    def dirk_step(self, t, y, h, args=()):
        """
        Usage: t, y, success = dirk_step(t, y, h, args)

        Utility routine to take a single diagonally-implicit RK time step of size h,
        where the inputs (t,y) are overwritten by the updated versions.
        args is used for optional parameters of the RHS.
        If success==True then the step succeeded; otherwise it failed.
        """

        # loop over stages, computing RHS vectors
        for i in range(self.s):

            # construct "data" for this stage solve
            self.data = np.copy(y)
            for j in range(i):
                self.data += h * self.A[i,j] * self.k[j,:]

            # solve the implicit stage (or copy the data for an explicit stage)
            tstage = t + h*self.c[i]
            if (abs(self.A[i,i]) > 1e-14):

                # construct implicit residual and Jacobian solver for this stage
                F = lambda zcur: zcur - self.data - h * self.A[i,i] * self.f(tstage, zcur, *args)
                self.sol.setup_linear_solver(tstage, -h * self.A[i,i], args)

                # perform implicit solve, and return on solver failure
                self.z, iters, success = self.sol.solve(F, y)
                self.nsol += 1
                if (not success):
                    return t, y, False
            else:
                self.z = self.data.copy()

            # store RHS at this stage
            self.k[i,:] = self.f(tstage, self.z, *args)

        # update time step solution
        for i in range(self.s):
            y += h * self.b[i] * self.k[i,:]
        t += h
        self.steps += 1
        return t, y, True

    def update_rhs(self, f):
        """ Updates the RHS function (cannot change vector dimensions) """
        self.f = f

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

        The fixed-step DIRK evolution routine

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
            raise ValueError("ERROR: DIRK::Evolve called without specifying a nonzero step size")

        # initialize output, and set first entry corresponding to initial condition
        y = y0.copy()
        Y = np.zeros((tspan.size,y0.size))
        Y[0,:] = y

        # initialize internal solution-vector-sized data
        self.k = np.zeros((self.s, y0.size), dtype=float)
        self.z = y0.copy()
        self.data = y0.copy()

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required, and the actual step size to use
            N = int(np.ceil((tspan[iout]-tspan[iout-1])/self.h))
            h = (tspan[iout]-tspan[iout-1]) / N

            # reset "current" t that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform diagonally-implicit Runge--Kutta update
                t, y, success = self.dirk_step(t, y, h, args)
                if (not success):
                    print("DIRK::Evolve error in time step at t =", t)
                    return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with "success" flag
        return Y, True

def Alexander3():
    """
    Usage: B = Alexander3()

    Utility routine to return the DIRK table corresponding to
    Alexander's 3-stage O(h^3) method.

    Reference: Alexander, SIAM J. Numer. Anal. 14 (1977), doi:10.1137/0714068.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    alpha = 0.43586652150845906
    tau2 = 0.5*(1.0+alpha)
    A = np.array((
        (alpha, 0.0, 0.0),
        (tau2-alpha, alpha, 0.0),
        (-0.25*(6.0*alpha*alpha - 16.0*alpha + 1.0), 0.25*(6.0*alpha*alpha - 20.0*alpha + 5.0), alpha)
    ), dtype=float)
    b = np.array((-0.25*(6.0*alpha*alpha - 16.0*alpha + 1.0),
                  0.25*(6.0*alpha*alpha - 20*alpha + 5.0), alpha))
    c = np.array((alpha, tau2, 1.0))
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def CrouzeixRaviart3():
    """
    Usage: B = CrouzeixRaviart3()

    Utility routine to return the DIRK table corresponding to
    Crouzeix & Raviart's 3-stage O(h^4) method.

    TODO: add citation

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    gamma = 1.0/np.sqrt(3.0)*np.cos(np.pi/18.0) + 0.5
    delta = 1.0/(6.0*(2.0*gamma-1.0)*(2.0*gamma-1.0))
    A = np.array((
        (gamma, 0.0, 0.0),
        (0.5-gamma, gamma, 0.0),
        (2.0*gamma, 1.0-4.0*gamma, gamma)
    ), dtype=float)
    b = np.array((delta, 1.0-2.0*delta, delta))
    c = np.array((gamma, 0.5, 1.0-gamma))
    p = 4
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def SDIRK5():
    """
    Usage: B = SDIRK5()

    Utility routine to return the SDIRK table corresponding to
    a 5-stage, 5th-order method.

    Reference: Kennedy & Carpenter, Diagonally implicit Runge--Kutta methods
               for ordinary differential equations. A review,
               NASA/TM-2016-219173 (2016).

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (4024571134387/14474071345096, 0, 0, 0, 0),
        (9365021263232/12572342979331, 4024571134387/14474071345096, 0, 0, 0),
        (2144716224527/9320917548702, -397905335951/4008788611757, 4024571134387/14474071345096, 0, 0),
        (-291541413000/6267936762551, 226761949132/4473940808273, -1282248297070/9697416712681, 4024571134387/14474071345096, 0),
        (-2481679516057/4626464057815, -197112422687/6604378783090, 3952887910906/9713059315593, 4906835613583/8134926921134, 4024571134387/14474071345096)
    ), dtype=float)
    b = np.array((-2522702558582/12162329469185, 1018267903655/12907234417901, 4542392826351/13702606430957, 5001116467727/12224457745473, 1509636094297/3891594770934))
    c = np.array((4024571134387/14474071345096, 5555633399575/5431021154178,  5255299487392/12852514622453, 3/20, 10449500210709/14474071345096))
    p = 5
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def EDIRK744():
    """
    Usage: B = EDIRK744()

    Utility routine to return the EDDIRK table corresponding to
    a 4th-order accurate method with semilinear order 4.

    TODO: add citation

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0, 0, 0, 0, 0),
        (66719178356146069/104971894986575178, 66719178356146069/104971894986575178, 0, 0, 0, 0, 0),
        (11574878994758291/117719113355115783, -1858197540898696/70529361366069153, 11617133062216757/43245479316548780, 0, 0, 0, 0),
        (312078294212599530/40823424700776821, 155312269009595199/86710391005988198, -743789150637775609/113352218631221311, 98271968880200657/179019545289054999, 0, 0, 0),
        (1246868775297421168/137070970121741807, 114921713922407255/52367417556902641, -205947502305419261/24454220481972685, 18936671640200689/104159855867653343, 47397311839212708/127463680130367391, 0, 0),
        (-151740509096074388/196613682401464609, 254369392774793867/44087509892864172, -73864359103986538/65744654972066205, -37706375961306427/179802732674457709, 7953265906419399/38933344132172515, 48325866641079469/46020097947328612, 0),
        (3312403043354842/33496693975407517, -7745264544994559/74509708869668763, 75463258779378077/134382831179297809, -11696764876217691/132584149662964151, 9114026243344448/106054923174086269, 55946924902076/75756035623695139, 34834932759942553/78271243704016222)
    ), dtype=float)
    b = A[-1,:]
    c = np.sum(A, axis=1)
    p = 4
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B
# Additional diagonally-implicit Runge--Kutta tables.

def IRK11():
    """
    Usage: B = IRK11()

    Utility routine to return the DIRK table corresponding to the IRK-1-1
    method.

    TODO: add citation

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array(((1,),), dtype=float)
    b = np.array((1,), dtype=float)
    c = np.array((1,), dtype=float)
    p = 1
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def Ascher232SDIRK():
    """
    Usage: B = Ascher232SDIRK()

    Utility routine to return the DIRK table corresponding to the
    Ascher(2,3,2)-SDIRK method.

    Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
               doi:10.1016/S0168-9274(97)00056-1.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0),
        (0, 0.29289321881345243, 0),
        (0, 0.70710678118654757, 0.29289321881345243)
    ), dtype=float)
    b = np.array((0, 0.70710678118654757, 0.29289321881345243), dtype=float)
    c = np.array((0, 0.29289321881345243, 1), dtype=float)
    p = 2
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def LobattoIIIA22IRK():
    """
    Usage: B = LobattoIIIA22IRK()

    Utility routine to return the DIRK table corresponding to the
    LobattoIIIA-2-2-IRK method.

    TODO: add citation

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0),
        (0.5, 0.5)
    ), dtype=float)
    b = np.array((0.5, 0.5), dtype=float)
    c = np.array((0, 1), dtype=float)
    p = 2
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def SSP222SDIRK():
    """
    Usage: B = SSP222SDIRK()

    Utility routine to return the DIRK table corresponding to the
    SSP2(2,2,2)-SDIRK method.

    Reference: Pareschi & Russo, J. Sci. Comput. 25 (2005),
               doi:10.1007/BF02728986.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.29289321881345254, 0),
        (0.41421356237309492, 0.29289321881345254)
    ), dtype=float)
    b = np.array((0.5, 0.5), dtype=float)
    c = np.array((0.29289321881345254, 0.70710678118654746), dtype=float)
    p = 2
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def SSP2332Lpm1SDIRK():
    """
    Usage: B = SSP2332Lpm1SDIRK()

    Utility routine to return the DIRK table corresponding to the
    SSP2(3,3,2)-lpm1-SDIRK method.

    Reference: Higueras, Happenhofer, Koch & Kupka, J. Comput. Appl. Math. 272
               (2014), doi:10.1016/j.cam.2014.05.011.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.18181818181818182, 0, 0),
        (0.30363851025008048, 0.18181818181818182, 0),
        (0.3465591182084175, 0.30434782608695654, 0.18181818181818182)
    ), dtype=float)
    b = np.array((0.33333333333333331, 0.33333333333333331, 0.33333333333333331), dtype=float)
    c = np.array((0.18181818181818182, 0.48545669206826231, 0.83272512611355587), dtype=float)
    p = 2
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def SSP2332Lpm2SDIRK():
    """
    Usage: B = SSP2332Lpm2SDIRK()

    Utility routine to return the DIRK table corresponding to the
    SSP2(3,3,2)-lpm2-SDIRK method.

    Reference: Higueras, Happenhofer, Koch & Kupka, J. Comput. Appl. Math. 272
               (2014), doi:10.1016/j.cam.2014.05.011.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.18181818181818182, 0, 0),
        (0.19406461307287753, 0.18181818181818182, 0),
        (0.28429036528210083, 0.47619047619047616, 0.18181818181818182)
    ), dtype=float)
    b = np.array((0.33333333333333331, 0.33333333333333331, 0.33333333333333331), dtype=float)
    c = np.array((0.18181818181818182, 0.37588279489105936, 0.94229902329075887), dtype=float)
    p = 2
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def SSP2332LpumSDIRK():
    """
    Usage: B = SSP2332LpumSDIRK()

    Utility routine to return the DIRK table corresponding to the
    SSP2(3,3,2)-lpum-SDIRK method.

    Reference: Higueras, Happenhofer, Koch & Kupka, J. Comput. Appl. Math. 272
               (2014), doi:10.1016/j.cam.2014.05.011.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.18181818181818182, 0, 0),
        (0.26623376623376621, 0.18181818181818182, 0),
        (0.34120425029515938, 0.34710743801652894, 0.18181818181818182)
    ), dtype=float)
    b = np.array((0.33333333333333331, 0.33333333333333331, 0.33333333333333331), dtype=float)
    c = np.array((0.18181818181818182, 0.44805194805194803, 0.87012987012987009), dtype=float)
    p = 2
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def SSP2332aDIRK():
    """
    Usage: B = SSP2332aDIRK()

    Utility routine to return the DIRK table corresponding to the
    SSP2(3,3,2)-a-DIRK method.

    Reference: Pareschi & Russo, J. Sci. Comput. 25 (2005),
               doi:10.1007/BF02728986.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.25, 0, 0),
        (0, 0.25, 0),
        (0.33333333333333331, 0.33333333333333331, 0.33333333333333331)
    ), dtype=float)
    b = np.array((0.33333333333333331, 0.33333333333333331, 0.33333333333333331), dtype=float)
    c = np.array((0.25, 0.25, 1), dtype=float)
    p = 2
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def SSP3332SDIRK():
    """
    Usage: B = SSP3332SDIRK()

    Utility routine to return the DIRK table corresponding to the
    SSP3(3,3,2)-SDIRK method.

    Reference: Pareschi & Russo, J. Sci. Comput. 25 (2005),
               doi:10.1007/BF02728986.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.29289321881345254, 0, 0),
        (0.41421356237309492, 0.29289321881345254, 0),
        (0.20710678118654746, 0, 0.29289321881345254)
    ), dtype=float)
    b = np.array((0.16666666666666666, 0.16666666666666666, 0.66666666666666663), dtype=float)
    c = np.array((0.29289321881345254, 0.70710678118654746, 0.5), dtype=float)
    p = 2
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def DBM53ESDIRK():
    """
    Usage: B = DBM53ESDIRK()

    Utility routine to return the DIRK table corresponding to the
    DBM-5-3-ESDIRK method.

    Reference: the DBM453 method of Vogl, Steyer, Reynolds, Ullrich &
               Woodward, J. Adv. Model. Earth Syst. 11 (2019),
               doi:10.1029/2019MS001700.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0, 0, 0),
        (-0.2228498531852541, 0.32591194130117246, 0, 0, 0),
        (-0.46801347074080546, 0.8634928422571696, 0.32591194130117246, 0, 0),
        (-0.46509906651927418, 0.81063103116959556, 0.6103672675683236, 0.32591194130117246, 0),
        (0.87795339639076675, -0.72692641526151547, 0.75204137157372719, -0.22898029400415087, 0.32591194130117246)
    ), dtype=float)
    b = np.array((0.87795339639076675, -0.72692641526151547, 0.75204137157372719, -0.2289802940041509, 0.32591194130117246), dtype=float)
    c = np.array((0, 0.1030620881159184, 0.72139131281753666, 1.2818111735198174, 1), dtype=float)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def Ascher233SDIRK():
    """
    Usage: B = Ascher233SDIRK()

    Utility routine to return the DIRK table corresponding to the
    Ascher(2,3,3)-SDIRK method.

    Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
               doi:10.1016/S0168-9274(97)00056-1.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0),
        (0, 0.78867513459481275, 0),
        (0, -0.57735026918962551, 0.78867513459481275)
    ), dtype=float)
    b = np.array((0, 0.5, 0.5), dtype=float)
    c = np.array((0, 0.78867513459481275, 0.21132486540518725), dtype=float)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def Ascher343SDIRK():
    """
    Usage: B = Ascher343SDIRK()

    Utility routine to return the DIRK table corresponding to the
    Ascher(3,4,3)-SDIRK method.

    Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
               doi:10.1016/S0168-9274(97)00056-1.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0, 0),
        (0, 0.435866521508459, 0, 0),
        (0, 0.28206673924577053, 0.435866521508459, 0),
        (0, 1.2084966491760101, -0.64436317068446924, 0.435866521508459)
    ), dtype=float)
    b = np.array((0, 1.2084966491760101, -0.64436317068446924, 0.435866521508459), dtype=float)
    c = np.array((0, 0.435866521508459, 0.71793326075422947, 1), dtype=float)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def Ascher443SDIRK():
    """
    Usage: B = Ascher443SDIRK()

    Utility routine to return the DIRK table corresponding to the
    Ascher(4,4,3)-SDIRK method.

    Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
               doi:10.1016/S0168-9274(97)00056-1.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.5, 0, 0, 0),
        (0.16666666666666666, 0.5, 0, 0),
        (-0.5, 0.5, 0.5, 0),
        (1.5, -1.5, 0.5, 0.5)
    ), dtype=float)
    b = np.array((1.5, -1.5, 0.5, 0.5), dtype=float)
    c = np.array((0.5, 0.66666666666666663, 0.5, 1), dtype=float)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def Cooper4ESDIRK():
    """
    Usage: B = Cooper4ESDIRK()

    Utility routine to return the DIRK table corresponding to the
    Cooper4-ESDIRK method.

    Reference: the order-3 methods with mu = 1/2 in Cooper & Sayfy, Math.
               Comp. 40 (1983), doi:10.1090/S0025-5718-1983-0679441-1.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0, 0),
        (-0.12200846792814619, 0.78867513459481275, 0, 0),
        (0.56100423396407306, -0.6830127018922193, 0.78867513459481275, 0),
        (0.25, 0.25, 0.5, 0)
    ), dtype=float)
    b = np.array((0.25, 0.25, 0.5, 0), dtype=float)
    c = np.array((0, 0.66666666666666663, 0.66666666666666663, 1), dtype=float)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def SSP3433SDIRK():
    """
    Usage: B = SSP3433SDIRK()

    Utility routine to return the DIRK table corresponding to the
    SSP3(4,3,3)-SDIRK method.

    Reference: Pareschi & Russo, J. Sci. Comput. 25 (2005),
               doi:10.1007/BF02728986.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.24169426078820999, 0, 0, 0),
        (-0.24169426078820999, 0.24169426078820999, 0, 0),
        (0, 0.75830573921179001, 0.24169426078820999, 0),
        (0.06042356519705, 0.1291528696059, 0.068729304408840008, 0.24169426078820999)
    ), dtype=float)
    b = np.array((0, 0.16666666666666666, 0.16666666666666666, 0.66666666666666663), dtype=float)
    c = np.array((0.24169426078820999, 0, 1, 0.5), dtype=float)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def EDIRK33():
    """
    Usage: B = EDIRK33()

    Utility routine to return the DIRK table corresponding to the EDIRK-3-3
    method.

    Reference: Sect. 3.2.3 of Conde, Gottlieb, Grant & Shadid, J. Sci. Comput.
               73 (2017), doi:10.1007/s10915-017-0560-2.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0),
        (0, 1, 0),
        (0.16666666666666666, -0.33333333333333331, 0.66666666666666663)
    ), dtype=float)
    b = np.array((0.16666666666666666, 0.16666666666666666, 0.66666666666666663), dtype=float)
    c = np.array((0, 1, 0.5), dtype=float)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def ESDIRK33():
    """
    Usage: B = ESDIRK33()

    Utility routine to return the DIRK table corresponding to the ESDIRK-3-3
    method.

    Reference: Sect. 3.2.3 of Conde, Gottlieb, Grant & Shadid, J. Sci. Comput.
               73 (2017), doi:10.1007/s10915-017-0560-2.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0),
        (0.21132486540518713, 0.78867513459481264, 0),
        (0.052831216351296784, -0.34150635094610965, 0.78867513459481287)
    ), dtype=float)
    b = np.array((0.16666666666666666, 0.16666666666666666, 0.66666666666666663), dtype=float)
    c = np.array((0, 1, 0.5), dtype=float)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def SDIRK45L1SA():
    """
    Usage: B = SDIRK45L1SA()

    Utility routine to return the DIRK table corresponding to the
    SDIRK4()5L[1]SA method.

    Reference: Table 22 of Kennedy & Carpenter, Diagonally implicit
               Runge--Kutta methods for ordinary differential equations. A
               review, NASA/TM-2016-219173 (2016).

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.25, 0, 0, 0, 0),
        (-0.10355339059327379, 0.25, 0, 0, 0),
        (-0.21875952311955699, 0.56177680358259996, 0.25, 0, 0),
        (-1.844429060379561, 1.8085200959895045, 0.88239716972389048, 0.25, 0),
        (0, 0.35033718067988195, 0.47575986600642217, -0.07609704668630414, 0.25)
    ), dtype=float)
    b = np.array((0, 0.35033718067988195, 0.47575986600642217, -0.07609704668630414, 0.25), dtype=float)
    c = np.array((0.25, 0.14644660940672621, 0.59301728046304292, 1.0964882053338338, 1), dtype=float)
    p = 4
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def LobattoIII34IRK():
    """
    Usage: B = LobattoIII34IRK()

    Utility routine to return the DIRK table corresponding to the
    LobattoIII-3-4-IRK method.

    TODO: add citation

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0),
        (0.25, 0.25, 0),
        (0, 1, 0)
    ), dtype=float)
    b = np.array((0.16666666666666666, 0.66666666666666663, 0.16666666666666666), dtype=float)
    c = np.array((0, 0.5, 1), dtype=float)
    p = 4
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def Cooper6ESDIRK():
    """
    Usage: B = Cooper6ESDIRK()

    Utility routine to return the DIRK table corresponding to the
    Cooper6-ESDIRK method.

    Reference: the first of the order-4 methods in Cooper & Sayfy, Math. Comp.
               40 (1983), doi:10.1090/S0025-5718-1983-0679441-1.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0, 0, 0, 0),
        (-0.5685790213016289, 1.0685790213016289, 0, 0, 0, 0),
        (0.25, -0.8185790213016289, 1.0685790213016289, 0, 0, 0),
        (0.25, 0.5342895106508144, -1.3528685319524434, 1.0685790213016289, 0, 0),
        (0, -2.1371580426032577, 4.442565331935536, -1.3054072893322786, 0, 0),
        (0.16666666666666666, 0, 0, 0.66666666666666663, 0.16666666666666666, 0)
    ), dtype=float)
    b = np.array((0.16666666666666666, 0, 0, 0.66666666666666663, 0.16666666666666666, 0), dtype=float)
    c = np.array((0, 0.5, 0.5, 0.5, 1, 1), dtype=float)
    p = 4
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B


def WSO32():
    """
    Usage: B = WSO32()

    Utility routine to return the DIRK table corresponding to the 4-stage,
    third-order, WSO-2 L-stable DIRK method.

    Reference: Sect. 3 of Ketcheson, Seibold, Shirokoff & Zhou, DIRK Schemes
               with High Weak Stage Order, Lecture Notes in Computational
               Science and Engineering (2020),
               doi:10.1007/978-3-030-39647-3_36.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.01900072890, 0.0, 0.0, 0.0),
        (0.40434605601, 0.38435717512, 0.0, 0.0),
        (0.06487908412, -0.16389640295, 0.51545231222, 0.0),
        (0.02343549374, -0.41207877888, 0.96661161281, 0.42203167233)
    ), dtype=float)
    b = A[-1, :].copy()
    c = A @ np.ones(4)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B


def WSO33():
    """
    Usage: B = WSO33()

    Utility routine to return the DIRK table corresponding to the 4-stage,
    third-order, WSO-3 L-stable DIRK method.

    Reference: Sect. 3 of Ketcheson, Seibold, Shirokoff & Zhou, DIRK Schemes
               with High Weak Stage Order, Lecture Notes in Computational
               Science and Engineering (2020),
               doi:10.1007/978-3-030-39647-3_36.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.13756543551, 0.0, 0.0, 0.0),
        (0.56695122794, 0.23483888782, 0.0, 0.0),
        (-1.08354072813, 2.96618223864, 0.44915521951, 0.0),
        (0.59761291500, -0.43420997584, -0.05305815322, 0.88965521406)
    ), dtype=float)
    b = A[-1, :].copy()
    c = A @ np.ones(4)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B


def WSO43():
    """
    Usage: B = WSO43()

    Utility routine to return the DIRK table corresponding to the 6-stage,
    fourth-order, WSO-3 L-stable DIRK method.

    Reference: Sect. 3 of Ketcheson, Seibold, Shirokoff & Zhou, DIRK Schemes
               with High Weak Stage Order, Lecture Notes in Computational
               Science and Engineering (2020),
               doi:10.1007/978-3-030-39647-3_36.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0.079672377876931, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.328355391763968, 0.136009256546967, 0.0, 0.0, 0.0, 0.0),
        (-0.650772774016417, 1.742859063495349, 0.256472952467792, 0.0, 0.0, 0.0),
        (-0.714580550967259, 1.793745752775934, -0.078254785672497, 0.311753794172585, 0.0, 0.0),
        (-1.120092779092918, 1.983452339867353, 3.117393885836001, -3.761930177913743, 0.770646024799205, 0.0),
        (0.214823667785537, 0.536367363903245, 0.154488125726409, -0.217748592703941, 0.072226422925896, 0.239843012362853)
    ), dtype=float)
    b = A[-1, :].copy()
    c = A @ np.ones(6)
    p = 4
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B


# Additional diagonally-implicit tables used by ARK pairs in the lecture notes.

def Ascher111SDIRK():
    """
    Usage: B = Ascher111SDIRK()

    Utility routine to return the padded SDIRK table corresponding to
    the implicit component of the ARS(1,1,1) method.

    Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
               doi:10.1016/S0168-9274(97)00056-1.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0),
        (0, 1)
    ), dtype=float)
    b = np.array((0, 1), dtype=float)
    c = np.array((0, 1), dtype=float)
    p = 1
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B


def Ascher122SDIRK():
    """
    Usage: B = Ascher122SDIRK()

    Utility routine to return the padded SDIRK table corresponding to
    the implicit component of the ARS(1,2,2) method.

    Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
               doi:10.1016/S0168-9274(97)00056-1.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0),
        (0, 0.5)
    ), dtype=float)
    b = np.array((0, 1), dtype=float)
    c = np.array((0, 0.5), dtype=float)
    p = 2
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B


def Ascher443PaddedSDIRK():
    """
    Usage: B = Ascher443PaddedSDIRK()

    Utility routine to return the padded SDIRK table corresponding to
    the implicit component of the ARS(4,4,3) method.

    Reference: Ascher, Ruuth & Spiteri, Appl. Numer. Math. 25 (1997),
               doi:10.1016/S0168-9274(97)00056-1.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0, 0, 0),
        (0, 0.5, 0, 0, 0),
        (0, 0.16666666666666666, 0.5, 0, 0),
        (0, -0.5, 0.5, 0.5, 0),
        (0, 1.5, -1.5, 0.5, 0.5)
    ), dtype=float)
    b = np.array((0, 1.5, -1.5, 0.5, 0.5), dtype=float)
    c = np.array((0, 0.5, 0.66666666666666663, 0.5, 1), dtype=float)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B


def ARKCouplingESDIRK3():
    """
    Usage: B = ARKCouplingESDIRK3()

    Utility routine to return the ESDIRK table corresponding to
    the third-order ESDIRK method from the ARK coupling example.

    TODO: add citation

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((
        (0, 0, 0, 0),
        (1/6, 1/3, 0, 0),
        (0.5, -1/3, 1/3, 0),
        (-2/3, 2/3, 2/3, 1/3)
    ), dtype=float)
    b = np.array((1/6, 0, 2/3, 1/6), dtype=float)
    c = np.array((0, 0.5, 0.5, 1), dtype=float)
    p = 3
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B
