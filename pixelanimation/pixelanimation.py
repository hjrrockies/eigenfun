import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from imageio import imread
import sys
from progress.bar import IncrementalBar

def grayscale_to_coords(image):
    rot_image = np.rot90(image,k=-1)
    rows,cols = np.unravel_index(np.argsort(rot_image,axis=None),shape=rot_image.shape)
    colors = np.sort(image.flatten())
    return rows+.5,cols+.5,colors

def animate_pixels(img1,img2,filename,verbose=False):
    if verbose: bar1 = IncrementalBar("Sorting", max=2,suffix='%(percent)d%%')
    rows1,cols1,colors1 = grayscale_to_coords(img1)
    if verbose: bar1.next()
    rows2,cols2,colors2 = grayscale_to_coords(img2)
    if verbose: bar1.next(); bar1.finish()

    aspect_ratio1 = img1.shape[0]/img1.shape[1]
    aspect_ratio2 = img2.shape[0]/img2.shape[1]
    plt.ioff()
    fig = plt.figure(figsize=(6.4,max(aspect_ratio1,aspect_ratio2)*6.4))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    plt.axis("off")
    plt.xlim((0,max(img1.shape[1],img2.shape[1])))
    plt.ylim((0,max(img1.shape[0],img2.shape[0])))
    pixels = img1.shape[1]
    pixels_per_inch = pixels/6.4
    size = 72/pixels_per_inch
    points = ax.scatter(rows1,cols1,c=colors1,cmap="gray",marker='s',s=size**2,vmin=0,vmax=1)

    n=100
    buffer = 10
    if verbose: bar2 = IncrementalBar("Interpolating",max=4,suffix='%(percent)d%%')
    colors = np.linspace(colors1,colors2,n)
    if verbose: bar2.next()
    rows = np.linspace(rows1,rows2,n)
    if verbose: bar2.next()
    cols = np.linspace(cols1,cols2,n)
    if verbose: bar2.next()
    pos = np.dstack((rows,cols))
    if verbose: bar2.next(); bar2.finish()
    total = 2*n+4*buffer
    def update(j):
        if j >= buffer and j < buffer+n:
            i = j-buffer
            points.set_offsets(pos[i])
            points.set_array(colors[i])
        elif j >= 3*buffer+n and j < 3*buffer+2*n:
            i = n-(j-(3*buffer+n))-1
            points.set_offsets(pos[i])
            points.set_array(colors[i])
        if verbose: bar3.next()

    if verbose: bar3 = IncrementalBar("Rendering",max=total,suffix='%(percent)d%%')
    ani = animation.FuncAnimation(fig,update,frames=total,interval=60)
    ani.save(filename)
    if verbose: bar3.next(); bar3.finish()
    plt.close(fig)
    plt.ion()
