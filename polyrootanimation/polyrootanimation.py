import numpy as np
import yroots as yr
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import os
from progress.bar import IncrementalBar
# import beampy as bp
# doc = bp.document()

# To let Bempy search automatically for a program replace
# the path by "auto" (check the default_theme.py file)

# doc._theme['document']['external_app'] = {"dvisvgm": "auto",}

plt.style.use('seaborn')

Writer = ani.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

def eval_homotopy(P,Q,T,X,Y,verbose=False):
    p1, p2 = P
    q1, q2 = Q
    if 'MultiCheb' in str(type(p1)):
        MultiX = yr.MultiCheb
    else: MultiX = yr.MultiPower
    pts = np.array([X.flatten(),Y.flatten()]).T
    arr1 = np.empty((len(T),*X.shape))
    arr2 = np.empty((len(T),*X.shape))
    roots = []
    xmin,xmax,ymin,ymax = X.min(),X.max(),Y.min(),Y.max()
    if verbose: bar1 = IncrementalBar("Computing homotopy\t", max=len(T),suffix='%(percent)d%%')
    for i,t in enumerate(T):
        coeff1 = (1-t)*p1.coeff+t*q1.coeff
        poly1 = MultiX(coeff1)
        coeff2 = (1-t)*p2.coeff+t*q2.coeff
        poly2 = MultiX(coeff2)
        arr1[i] = poly1(pts).reshape(X.shape)
        arr2[i] = poly2(pts).reshape(X.shape)
        roots.append(rootfilter(yr.polysolve([poly1,poly2]),xmin,xmax,ymin,ymax))
        if verbose: bar1.next()
    if verbose: bar1.finish()
    return arr1,arr2,roots,T

def rootfilter(roots,xmin=1.1,xmax=1.1,ymin=1.1,ymax=1.1):
    realmask = (np.abs(roots[:,0].imag)<1e-16)&(np.abs(roots[:,1].imag)<1e-16)
    roots = roots[realmask].real
    intervalmask = (roots[:,0]>=xmin)&(roots[:,0]<=xmax)&(roots[:,1]>=ymin)&(roots[:,1]<=ymax)
    roots = roots[intervalmask]
    return roots

def gen_polynomials(deg,power):
    p1 = yr.polynomial.getPoly(deg,2,power)
    p2 = yr.polynomial.getPoly(deg,2,power)
    q1 = yr.polynomial.getPoly(deg,2,power)
    q2 = yr.polynomial.getPoly(deg,2,power)
    return [p1,p2],[q1,q2]

def plot_start_end(P,Q,X,Y):
    p1,p2 = P
    q1,q2 = Q
    pts = np.array([X.flatten(),Y.flatten()]).T
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(121)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.contour(X,Y,p1(pts).reshape(X.shape),levels=[0],colors='C0')
    ax.contour(X,Y,p2(pts).reshape(X.shape),levels=[0],colors='C1')
    # plt.scatter(roots[:,0],roots[:,1],c='k',s=15,zorder=3)
    ax.axis('tight')
    plt.axis('off')
    ax.set_aspect('equal')
    ax = plt.subplot(122)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.contour(X,Y,q1(pts).reshape(X.shape),levels=[0],colors='C2')
    ax.contour(X,Y,q2(pts).reshape(X.shape),levels=[0],colors='C4')
    # plt.scatter(roots[:,0],roots[:,1],c='k',s=15,zorder=3)
    ax.axis('tight')
    plt.axis('off')
    ax.set_aspect('equal')
    plt.show()

def sigmoid_homotopy(P,Q,X,Y,verbose=False):
    t = np.linspace(0,1,300)
    beta = 2
    f = lambda t: 1/(1+(t/(1-t))**(-beta))
    T = f(t)
    T[0],T[-1] = 0,1
    return eval_homotopy(P,Q,T,X,Y,verbose)

def animate_homotopy(P,Q,X,Y,filename=None,c1='C0',c2='C1',c='white',facecolor='black',verbose=False):
    if facecolor == 'white': c='black'
    arr1,arr2,roots,T = sigmoid_homotopy(P,Q,X,Y,verbose)

    fig = plt.figure(figsize=(6,6),dpi=200)
    ax = plt.gca()
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.grid(False)
    ax.set_aspect('equal')
    ax.set_facecolor(facecolor)

    # contours = [ax.contour(X,Y,arr1[0],levels=[0],colors=c1),
    #             ax.contour(X,Y,arr2[0],levels=[0],colors=c2)]
    scatter = plt.scatter(roots[0][:,0],roots[0][:,1],c=c,s=15,zorder=3)

    def update(i):
        global contours
        if i < len(T): j = i
        elif i >= len(T): j = len(T) - i - 1
        try:
            for contour in contours:
                for obj in contour.collections:
                    obj.remove()
        except:
            pass
        contours = [ax.contour(X,Y,arr1[j],levels=[0],colors=c1),
                    ax.contour(X,Y,arr2[j],levels=[0],colors=c2)]
        scatter.set_offsets(roots[j])
        if verbose: bar2.next()

    if verbose: bar2 = IncrementalBar("Rendering\t\t",max=2*len(T),suffix='%(percent)d%%')

    animation = ani.FuncAnimation(fig,update,frames=2*len(T),interval=20)

    if filename is None:
        i = 0
        while os.path.exists(f"homotopy{i}.mp4"):
            i += 1
        filename = f"homotopy{i}.mp4"
    animation.save(filename,writer=writer)
    if verbose:
        bar2.finish()
        print(f'saving as {filename}')

# def animate_homotopy_html(P,Q,X,Y,filename=None,c1='C0',c2='C1',c='white',facecolor='black',verbose=False):
#     if facecolor == 'white': c='black'
#     arr1,arr2,roots,T = sigmoid_homotopy(P,Q,X,Y,verbose)
#
#     with bp.slide(''):
#         anim_figs = []
#         for i in range(2*len(T)):
#             if i < len(T): j = i
#             elif i >= len(T): j = len(T) - i - 1
#             fig = plt.figure(figsize=(6,6),dpi=200)
#             ax = plt.gca()
#             fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
#             ax.grid(False)
#             ax.set_aspect('equal')
#             ax.set_facecolor(facecolor)
#             plt.scatter(roots[j][:,0],roots[j][:,1],c=c,s=15,zorder=3)
#             ax.contour(X,Y,arr1[j],levels=[0],colors=c1)
#             ax.contour(X,Y,arr2[j],levels=[0],colors=c2)
#             plt.close(fig)
#             anim_figs.append(fig)
#
#         bp.animatesvg(anim_figs)
#     bp.save('polyroot.html')
