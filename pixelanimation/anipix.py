USAGE = """USAGE

$ python anipix.py [imfile1] [imfile2] [outfile] [--color]

[outfile] must be .mp4

[--c] is an optional flag to use color mode (slower)

Examples:

    $ python anipix.py camera2.png lena.png mixing.mp4

    $ python anipix.py im1.jpg im2.jpg mixing2.mp4 --c
"""


if __name__ == "__main__":
    from sys import argv
    import pixelanimation as pa

    if len(argv) < 4 or "--help" in argv:
        print(USAGE)

    elif argv[3][-4:] != ".mp4":
        print("Bad input: [outfile] must be .mp4\n")
        print(USAGE)

    else:
        if "--c" in argv:
            print("Using color mode")
            pa.animate_pixels(argv[1],argv[2],argv[3],verbose=True,color=True)
        else:
            pa.animate_pixels(argv[1],argv[2],argv[3],verbose=True)
