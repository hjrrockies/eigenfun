import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def SIRD(t_eval,I0,beta,gamma,delta1,delta2,K,severe):
    y0 = np.array([1-I0,I0,0,0])
    def func(t,y,beta,gamma,delta1,delta2,K,severe):
        if severe*y[1] <= K:
            return np.array([-beta*y[0]*y[1],beta*y[0]*y[1]-gamma*y[1],gamma*(1-delta1)*y[1],gamma*delta1*y[1]])
        else:
            frac = (delta1*K+delta2*(y[1]-K))/y[1]
            return np.array([-beta*y[0]*y[1],beta*y[0]*y[1]-gamma*y[1],gamma*(1-frac)*y[1],gamma*frac*y[1]])
    return solve_ivp(func,(t_eval[0],t_eval[-1]),y0,t_eval=t_eval,args=(beta,gamma,delta1,delta2,K,severe)).y

def plot(I0,beta,gamma,delta1,delta2,K):
    t_eval = np.arange(365)
    y = SIRD(t_eval,I0,beta,gamma,delta1,delta2,K)
    fig = plt.figure()
    labels = ['Susceptible','Infected','Recovered','Died']
    for i in range(4):
        plt.plot(t_eval,y[i],label=labels[i])
    plt.ylim(0,1)
    plt.legend()
    plt.show()
