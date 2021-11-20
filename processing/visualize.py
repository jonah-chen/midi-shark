from matplotlib import pyplot as plt
import numpy as np
import argparse

def visualize(filename):
    """
    Open a npy file and visualize the data.
    """
    data = np.load(filename)
    plt.figure(figsize=(8, 6), dpi=80)
    plt.imshow(data)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_file', help='.npy file to parse', type=str)
    args = parser.parse_args()
    visualize(args.npy_file)
