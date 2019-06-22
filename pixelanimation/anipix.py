if __name__ == "__main__":
    from sys import argv
    import pixelanimation as pa
    from imageio import imread
    import numpy as np

    img1 = np.array(imread(argv[1],as_gray=True))/256
    img2 = np.array(imread(argv[2],as_gray=True))/256

    if img1.shape[0]*img1.shape[1] != img2.shape[0]*img2.shape[1]:
        raise ValueError("Images must have the name number of pixels")
        
    pa.animate_pixels(img1,img2,argv[3],verbose=True)
