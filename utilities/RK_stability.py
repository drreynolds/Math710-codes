#!/usr/bin/env python
# Function to generate and plot the linear stability regions for Runge--Kutta
# methods.  Includes a simple "main" that uses this function to plot the
# stability region for forward and backward Euler (when posed as RK methods).
#
# Daniel R. Reynolds
# Math & Stat @ UMBC

# general imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('../04_explicit_one_step')
sys.path.append('../05_implicit_one_step')
from ERK import ERK4
from DIRK import CrouzeixRaviart3

def RK_stability_function(B, z):
    ''' Usage: R = RK_stability_function(B, z)

        Inputs:
          B is a Butcher table, with components:
             B['A'] -- the Butcher table matrix
             B['b'] -- the solution coefficients
          z is an array of points in the complex plane

        Outputs:
          R is an array of the same shape as z, holding the values of the
            RK stability function
              R(z) = 1 + z * b^T inv(I-z*A) e
            at each entry of z.  The linear systems for all entries are
            solved together, as a "stack" of s x s systems.  At an entry of
            z that lies exactly on a pole of R, R is set to infinity.'''

    # extract the components of the Butcher table
    A = np.asarray(B['A'], dtype=float)
    b = np.asarray(B['b'], dtype=float)
    s = b.size

    # solve (I - z*A) k = e at every entry of z at once
    zv = np.asarray(z, dtype=complex).ravel()
    M = np.eye(s)[None,:,:] - zv[:,None,None]*A[None,:,:]
    pole = np.zeros(zv.size, dtype=bool)
    try:
        K = np.linalg.solve(M, np.ones((zv.size, s, 1), dtype=complex))[:,:,0]
    except np.linalg.LinAlgError:
        # entries of z may land on a pole, where I - z*A is singular; there
        # we solve the systems one at a time, and mark the poles
        K = np.zeros((zv.size, s), dtype=complex)
        for n in range(zv.size):
            try:
                K[n,:] = np.linalg.solve(M[n], np.ones(s, dtype=complex))
            except np.linalg.LinAlgError:
                pole[n] = True
    with np.errstate(all='ignore'):   # R may overflow near a pole
        R = 1 + zv*(K @ b)
    R[pole] = np.inf
    return R.reshape(np.shape(z))

def RK_stability(B, box, N=1000):
    ''' Usage: X,Y = RK_stability(B, box, N)

        Inputs:
          B is a Butcher table, with components:
             B['A'] -- the Butcher table matrix
             B['b'] -- the solution coefficients
          box = [xl, xr, yl, yr] is the bounding box for the sub-region
              of the complex plane in which to perform the test
          N is optional, specifying how many sample sub-region points to use

        Outputs:
          (X, Y) where X is an array of real components of the stability boundary
            and Y is an array of imaginary components of the stability boundary

        We consider the RK stability function
          R(eta) = 1 + eta * dot(b, inv(I-eta*A)*e)

        We sample the values in 'box' within the complex plane, plugging
        each value into |R(eta)| (using RK_stability_function), and plot the
        contour of this function having value 1.'''
    import matplotlib.pyplot as pyplot
    import numpy as np

    # set mesh of sample points
    xl = box[0]
    xr = box[1]
    yl = box[2]
    yr = box[3]
    x = np.linspace(xl, xr, N)
    y = np.linspace(yl, yr, N)
    X, Y = np.meshgrid(x, y)

    # evaluate |R(eta)| for each eta in the mesh
    R = np.abs(RK_stability_function(B, X + Y*1j))

    # create contour
    eps = np.finfo(float).eps
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    plt.plot([box[0],box[1]],[0,0],'k--')
    plt.plot([0,0],[box[2],box[3]],'k--')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(box[0],box[1])
    ax.set_ylim(box[2],box[3])
    contour_set = pyplot.contourf(x, y, R, levels=(-eps,1.0))
    pyplot.contour(x, y, R, levels=(1.0,), colors='k', linewidths=1.25)

    # extract and return vertices in the contour R = 1
    # (ContourSet.collections was removed in matplotlib 3.10; since 3.8 the
    # ContourSet itself holds one path per contour level)
    mpl_version = tuple(int(v) for v in matplotlib.__version__.split('.')[:2])
    if (mpl_version >= (3,8)):
        vertices = contour_set.get_paths()[0].vertices
    else:
        vertices = contour_set.collections[0].get_paths()[0].vertices
    return vertices[:,0], vertices[:,1]


if __name__ == '__main__':
    ''' Driver that calls RK_stability to plot the stability regions for forward
        Euler, ERK4, backward Euler, and the Crouzeix & Raviart 3rd-order DIRK method.'''

    A = np.zeros((1,1))
    b = np.zeros(1)
    b[0] = 1.0
    B = {'A': A, 'b': b}
    box = [-3.0, 1.0, -2.0, 2.0]
    (x,y) = RK_stability(B, box, 100)
    plt.title('Forward Euler stability region (shaded = stable)')
    plt.savefig('FE_stability.png')

    box = [-5.0, 1.0, -3.0, 3.0]
    (x,y) = RK_stability(ERK4(), box, 100)
    plt.title('ERK4 stability region (shaded = stable)')
    plt.savefig('RK4_stability.png')

    A = np.zeros((1,1))
    A[0,0] = 1.0
    b = np.zeros(1)
    b[0] = 1.0
    B = {'A': A, 'b': b}
    box = [-1.0, 3.0, -2.0, 2.0]
    (x,y) = RK_stability(B, box, 100)
    plt.title('Backward Euler stability region (shaded = stable)')
    plt.savefig('BE_stability.png')

    box = [-10.0, 10.0, -10.0, 10.0]
    (x,y) = RK_stability(CrouzeixRaviart3(), box, 100)
    plt.title('CrouzeixRaviart3 stability region (shaded = stable)')
    plt.savefig('CR3_stability.png')

    plt.show()


# end of script
