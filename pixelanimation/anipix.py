USAGE = """USAGE

$ python anipix.py [imfile1] [imfile2] [outfile]

[outfile] must be .mp4

Example:

    $ python anipix.py camera2.png lena.png mixing.mp4
"""


if __name__ == "__main__":
    from sys import argv
    import pixelanimation as pa
    from imageio import imread
    import numpy as np

    if len(argv) < 4 or "--help" in argv:
        print(USAGE)

    elif argv[3][-4:] != ".mp4":
        print("Bad input: [outfile] must be .mp4\n")
        print(USAGE)

    else:
        img1 = np.array(imread(argv[1],as_gray=True))/256
        img2 = np.array(imread(argv[2],as_gray=True))/256

        if img1.shape[0]*img1.shape[1] != img2.shape[0]*img2.shape[1]:
            raise ValueError("Images must have the name number of pixels")

        pa.animate_pixels(img1,img2,argv[3],verbose=True)
