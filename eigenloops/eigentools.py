import numpy as np
import scipy.linalg as la
from progress.bar import IncrementalBar

def eig_trajectories(A,T,verbose=False):
    """Computes the trajectories of the eigenvalues of the
    matrix function A(t)

    Parameters
    ----------
    A : callable
        Matrix-valued function of one parameter t
    T : 1d array
        Values of the parameter t

    Returns
    -------
    E : ndarray
        Array of eigenvalue trajectories where E[i] is the
        trajectory of the ith eigenvalue as a 1d array
    """
    n,m = A(T[0]).shape
    if n!=m:
        raise ValueError("Matrix must be square")

    m = len(T)
    E = np.empty((n,m),dtype="complex")
    E[:,0] = la.eig(A(T[0]),right=False)
    if verbose: bar = IncrementalBar("Calculating\t", max=m,suffix='%(percent)d%%')
    for i,t in enumerate(T[1:]):
        w = la.eig(A(t),right=False)
        mask = list(range(n))
        for eig in w:
            idx = np.argmin(np.abs(eig-E[:,i][mask]))
            E[mask[idx],i+1] = eig
            del mask[idx]
        if verbose: bar.next()
    if verbose: bar.next(); bar.finish()
    return E

def eig_loops(A,U,V,verbose=False):
    """Computes the loops of eigenvalues for the matrix function A(u,v)

    Parameters
    ----------
    A : callable
        Matrix-valued function of two parameters u,v
    U : 1d array
        Values of the parameter u
    V : 1d array
        Values of the parameter v

    Returns
    -------
    L : ndarray
        Array of eigenvalue loops where L[i] is a 2d array for the ith eigenvalue.
        L[i,j,k] = the ith eigenvalue of A(U[j],V[k])
    """
    n,m = A(U[0],V[0]).shape
    if n!=m:
        raise ValueError("Matrix must be square")

    m = len(U)
    l = len(V)

    L = np.empty((n,m,l),dtype="complex")

    B = lambda u: A(u,V[0])
    L[:,:,0] = eig_trajectories(B,U)

    if verbose: bar = IncrementalBar("Calculating\t", max=m,suffix='%(percent)d%%')
    for i,v in enumerate(V[1:]):
        B = lambda u: A(u,v)
        E = eig_trajectories(B,U)
        mask = list(range(n))
        for traj in E:
            idx = np.argmin(np.abs(traj[0]-L[:,0,i][mask]))
            L[mask[idx],:,i+1] = traj
            del mask[idx]
        if verbose: bar.next()
    if verbose: bar.next(); bar.finish()
    return L

def eigenvector_trajectories(A,T,verbose=False):
    """Computes the trajectories of the eigenvalues and eigenvectors of the
    matrix-valued function A(t)

    Parameters
    ----------
    A : callable
        Matrix-valued function of one parameter t
    T : 1d array
        Values of the parameter t

    Returns
    -------
    E : ndarray
        Array of eigenvalue trajectories where E[i] is the
        trajectory of the ith eigenvalue as a 1d array
    V : ndarray
        Array of eigenvector trajectories where V[i] is the trajectory of the ith
        eigenvector. V[:,i,k] = ith eigenvector of A(T[k])
    """

    n,m = A(T[0]).shape
    if n!=m:
        raise ValueError("Matrix must be square")

    m = len(T)
    E = np.empty((n,m),dtype="complex")
    V = np.empty((n,n,m),dtype="complex")
    E[:,0], V[:,:,0] = la.eig(A(T[0]))
    if verbose: bar = IncrementalBar("Calculating\t", max=m,suffix='%(percent)d%%')
    for i,t in enumerate(T[1:]):
        w,v = la.eig(A(t))
        mask = list(range(n))
        for eig in w:
            idx = np.argmin(np.abs(eig-E[:,i][mask]))
            E[mask[idx],i+1] = eig
            V[:,mask[idx],i+1] = v[:,mask[idx]]*np.sign(v[:,mask[idx]]@V[:,mask[idx],i])
            del mask[idx]
        if verbose: bar.next()
    if verbose: bar.next(); bar.finish()
    return E,V
