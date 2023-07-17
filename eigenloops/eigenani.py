import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from progress.bar import IncrementalBar
import eigentools as et

def animate_eig(A,T,outfile,verbose=False):
    """Animates the eigenvalues of the matrix function A(t). Saves the animation
    as "outfile".

    Parameters
    ----------
    A : callable
        Matrix-valued function of one parameter t
    T : 1d array
        Values of the parameter t
    """
    n = A(T[0]).shape[0]
    E = et.eig_trajectories(A,T,verbose=verbose)

    #set up figure
    fig = plt.figure(figsize=(6,6),dpi=100)
    ax = plt.gca()
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.grid(False)
    ax.set_facecolor('black')
    x0,x1 = E.real.min()-1,E.real.max()+1
    y0,y1 = E.imag.min()-1,E.imag.max()+1
    plt.xlim((x0,x1))
    plt.ylim((y0,y1))

    #create line objects
    if n <= 10:
        points = [plt.plot([],[],c=f'C{k}',marker='o')[0] for k in range(n)]
        trajectories = [plt.plot([],[],c=f'C{k}')[0] for k in range(n)]
    else:
        points = [plt.plot([],[],c='C0',marker='o')[0] for k in range(n)]
        trajectories = [plt.plot([],[],c='C0')[0] for k in range(n)]

    #function to update line objects
    def update(i):
        for j in range(n):
            points[j].set_data(E[j,i].real,E[j,i].imag)
            trajectories[j].set_data(E[j,:i+1].real,E[j,:i+1].imag)
        if verbose: bar.next()

    #animation
    if verbose: bar = IncrementalBar("Rendering\t",max=len(T),suffix='%(percent)d%%')
    ani = animation.FuncAnimation(fig,update,frames=len(T),interval=15)
    ani.save(outfile)
    if verbose: bar.next(), bar.finish()

def animate_eig_loops(A,U,V,outfile,verbose=False):
    """Animates the loops of eigenvalues for the matrix function A(u,v). Saves
    the animation as "outfile".

    Parameters
    ----------
    A : callable
        Matrix-valued function of two parameters u,v
    U : 1d array
        Values of the parameter u
    V : 1d array
        Values of the parameter v
    """
    n = A(U[0],V[0]).shape[0]
    L = et.eig_loops(A,U,V,verbose=verbose)

    #set up figure
    # plt.ioff()
    fig = plt.figure(figsize=(6,6),dpi=100)
    ax = plt.gca()
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.grid(False)
    # ax.set_aspect('equal')
    ax.set_facecolor('black')
    x0,x1 = L.real.min()-1,L.real.max()+1
    y0,y1 = L.imag.min()-1,L.imag.max()+1
    plt.xlim((x0,x1))
    plt.ylim((y0,y1))

    #create line objects
    trajectories = [plt.plot([],[],c='C5')[0] for k in range(n)]

    #function to update line objects
    def update(i):
        for j in range(n):
            trajectories[j].set_data(L[j,:,i].real,L[j,:,i].imag)
        if verbose: bar.next()

    #animation
    if verbose: bar = IncrementalBar("Rendering\t",max=len(V),suffix='%(percent)d%%')
    ani = animation.FuncAnimation(fig,update,frames=len(V),interval=50)
    ani.save(outfile)
    if verbose: bar.next(); bar.finish()
    # plt.ion()
