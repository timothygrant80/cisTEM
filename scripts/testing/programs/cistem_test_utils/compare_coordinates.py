#!/usr/bin/env python3

import numpy as np
from os.path import join
import argparse


def print_basic_stats(coords_1_dir, coords_2_dir):
    coords_1 = join(coords_1_dir, 'coordinates.txt')
    coords_2 = join(coords_2_dir, 'coordinates.txt')
    # Read in the coordinates using numpy from the text files, ignoring lines that start with #
    coords_1 = np.loadtxt(coords_1, comments='#')
    coords_2 = np.loadtxt(coords_2, comments='#')

    print("There are {} coordinates in the first file".format(len(coords_1)))
    print("There are {} coordinates in the second file".format(len(coords_2)))

    # Get the average of the peak values in column 8
    avg_1 = np.average(coords_1[:, 7])
    avg_2 = np.average(coords_2[:, 7])

    # print out the average to three decimal places
    print("The average peak value in the first file is {:.3f}".format(avg_1))
    print("The average peak value in the second file is {:.3f}".format(avg_2))


if __name__ == '__main__':

    # Parse the args with argparse, we expect two directories as the first and second arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('coords_1_dir', help='Path to the first directory')
    parser.add_argument('coords_2_dir', help='Path to the second directory')
    args = parser.parse_args()

    print_basic_stats(args.coords_1_dir, args.coords_2_dir)
