from sys import argv
import numpy as np
import eigenani as ea

if __name__ == "__main__":
    n = int(argv[1])
    M1,M2,M3 = np.random.randn(3,n,n)+1j*np.random.randn(3,n,n)
    A = lambda u,v: M1 + .2*np.exp(1j*u)*M2 + .1*np.exp(1j*v)*M3
    U = np.linspace(0,2*np.pi,300)
    ea.animate_eig_loops(A,U,U,argv[2],verbose=True)
