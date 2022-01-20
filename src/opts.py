'''
Hyperparameters wrapped in argparse
This file contains most of tuanable parameters for this homework


You can change the values by changing their default fields or by command-line
arguments. For example, "python q2_1_4.py --sigma 0.15 --ratio 0.7"
'''

import argparse
import numpy as np

def get_opts():
    parser = argparse.ArgumentParser(description='16-720 HW2: Homography')

    ## Feature detection (requires tuning)
    parser.add_argument('--sigma', type=float, default=0.15,
                        help='threshold for corner detection using FAST feature detector')
    parser.add_argument('--ratio', type=float, default=0.7,
                        help='ratio for BRIEF feature descriptor')

    ## Ransac (requires tuning)
    parser.add_argument('--max_iters', type=int, default=1000,
                        help='the number of iterations to run RANSAC for')
    parser.add_argument('--inlier_tol', type=float, default=5.0,
                        help='the tolerance value for considering a point to be an inlier')

    ## Additional options (add your own hyperparameters here)
    parser.add_argument('--gui_size', type=int, default=512,
                        help='the size of the square video/image the user sees')
    parser.add_argument('--frame_width', type=int, default=1280,
                        help='the video frame width resolution')
    parser.add_argument('--frame_height', type=int, default=720,
                        help='the video frame height resolution')
    parser.add_argument('--num_matches', type=int, default=8,
                        help='the number of feature matches for calibrating the homography')
    parser.add_argument('--k_means_segments', type=int, default=4,
                        help='the number of segments for the kmean segmentation')
    parser.add_argument('--grid_size', type=int, default=8,
                        help='the size of the go board grid')
    parser.add_argument('--red', type=int, default=np.array([0, 0, 255]),
                        help='the color red for bgr convention')
    parser.add_argument('--black', type=int, default=np.array([0, 0, 0]),
                        help='the color black for bgr convention')
    parser.add_argument('--white', type=int, default=np.array([255, 255, 255]),
                        help='the color white for bgr convention')
    ##
    opts = parser.parse_args()

    return opts
