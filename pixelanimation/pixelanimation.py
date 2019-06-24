import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import rgb_to_hsv
from imageio import imread
from progress.bar import IncrementalBar

def grayscale_to_coords(image):
    """Sorts a grayscale image's pixels by saturation, and returns arrays
    containing the row and column positions corresponding to sorted order,
    as well as the sorted intensities as a flattened array."""
    rot_image = np.rot90(image,k=-1)
    rows,cols = np.unravel_index(np.argsort(rot_image,axis=None),shape=rot_image.shape)
    colors = np.sort(rot_image.flatten())
    return rows,cols,colors

def color_to_coords(image):
    """Sorts a color image's pixels by hue, and returns arrays containing the
    row and column positions corresponding to sorted order, as well as the
    sorted rgb values as a Nx3 array."""
    rot_image = np.rot90(image,k=-1)
    hue = rgb_to_hsv(rot_image)[:,:,0]
    mask = np.argsort(hue,axis=None)
    rows,cols = np.unravel_index(mask,shape=rot_image.shape)
    colors = rot_image.reshape((rot_image.shape[0]*rot_image.shape[1],3))[mask]
    return rows,cols,colors

def animate_pixels(imfile1,imfile2,outfile,color=False,verbose=False):
    """Animates a pixel-motion transition between two images. Images must have
    the exact same number of pixels. Animation is saved as "outfile".

    Parameters
    ----------
    imfile1 : str or file object
        The file name or file object for the first image
    imfile2 : str or file object
        The file name or file object for the second image
    outfile : str
        The output file name
    color : bool, optional
        If True, runs in color mode
    verbose : bool, optional
        If True, displays a progress bar in the console
    """

    # Read in images
    if color:
        img1 = np.array(imread(imfile1))/255
        img2 = np.array(imread(imfile2))/255
    else:
        img1 = np.array(imread(imfile1,as_gray=True))/255
        img2 = np.array(imread(imfile2,as_gray=True))/255

    # Check number of pixels
    if img1.shape[0]*img1.shape[1] != img2.shape[0]*img2.shape[1]:
        raise ValueError("Images must have the name number of pixels")

    # Sort pixels by saturation (if grayscale) or hue (if color)
    if verbose: bar1 = IncrementalBar("Sorting", max=2,suffix='%(percent)d%%')
    if color: rows1,cols1,colors1 = color_to_coords(img1)
    else: rows1,cols1,colors1 = grayscale_to_coords(img1)
    if verbose: bar1.next()
    if color: rows2,cols2,colors2 = color_to_coords(img2)
    else: rows2,cols2,colors2 = grayscale_to_coords(img2)
    if verbose: bar1.next(); bar1.finish()

    # n is number of frames of one-directional transition
    # buffer is number of stationary frames before and after the transitions
    # total is number of frames for two transitions with 2 buffer periods each
    n=100
    buffer = 10
    total = 2*n+4*buffer

    # np.linspace creates evenly spaced position and color arrays for transition
    if verbose: bar2 = IncrementalBar("Interpolating",max=4,suffix='%(percent)d%%')
    colors = np.linspace(colors1,colors2,n)
    if verbose: bar2.next()
    rows = np.linspace(rows1+.5,rows2+.5,n)
    if verbose: bar2.next()
    cols = np.linspace(cols1+.5,cols2+.5,n)
    if verbose: bar2.next()
    pos = np.dstack((rows,cols))
    if verbose: bar2.next(); bar2.finish()

    # Calculate the aspect ratio of the two images
    aspect_ratio1 = img1.shape[0]/img1.shape[1]
    aspect_ratio2 = img2.shape[0]/img2.shape[1]

    plt.ioff()
    # Figure will always have default matplotlib 6.4 inch width
    fig = plt.figure(figsize=(6.4,max(aspect_ratio1,aspect_ratio2)*6.4))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    plt.axis("off")
    plt.xlim((0,max(img1.shape[1],img2.shape[1])))
    plt.ylim((0,max(img1.shape[0],img2.shape[0])))

    # Markers are measured in points, which are 1/72nd of an inch. Calculates
    # pixel size in points
    pixels = max(img1.shape[1],img2.shape[1])
    pixels_per_inch = pixels/6.4
    size = 72/pixels_per_inch

    # core object is a scatter plot with square markers set to pixel size
    if color:
        points = ax.scatter(rows[0],cols[0],c=colors1,marker='s',s=size**2)
    else:
        points = ax.scatter(rows[0],cols[0],c=colors1,cmap="gray",marker='s',s=size**2,vmin=0,vmax=1)

    # update function changes the scatter plot at each frame
    # set_color works for rgb, set_array works for grayscale
    def update(j):
        if j >= buffer and j < buffer+n:
            i = j-buffer
            points.set_offsets(pos[i])
            if color: points.set_color(colors[i])
            else: points.set_array(colors[i])
        elif j >= 3*buffer+n and j < 3*buffer+2*n:
            i = n-(j-(3*buffer+n))-1
            points.set_offsets(pos[i])
            if color: points.set_color(colors[i])
            else: points.set_array(colors[i])
        if verbose: bar3.next()

    if verbose: bar3 = IncrementalBar("Rendering",max=total,suffix='%(percent)d%%')

    # Create FuncAnimation with 60-millisecond inteval between frames
    ani = animation.FuncAnimation(fig,update,frames=total,interval=60)

    # Save animation and close the figure
    ani.save(outfile)
    if verbose: bar3.next(); bar3.finish()
    plt.close(fig)
    plt.ion()
