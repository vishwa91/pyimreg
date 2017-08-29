#!/usr/bin/env python

from scipy import *
from scipy.linalg import *
from scipy.special import *
from random import choice
from PIL import Image
import sys

from sift import *
from homography import *

import pdb

# New version coming soon.
def get_points(locs1, locs2, matchscores):
    '''
        Return the corresponding points in both the images
    '''
    plist = []
    t = min(len(locs1), len(locs2))
    for i in range(len(matchscores)):
        if (matchscores[i] > 0):
            y1 = int(locs1[i, 1])
            x1 = int(locs1[i, 0])

            y2 = int(locs2[int(matchscores[i]), 1])
            x2 = int(locs2[int(matchscores[i]), 0])

            plist.append([[x1,y1],[x2,y2]])
    return plist

def get_homography(points_list):
    '''
        Function to quickly compute a homography matrix from all point 
        correspondences.

        Inputs:
            points_list: tuple of tuple of tuple of correspondence indices. Each
            entry is [[x1, y1], [x2, y2]] where [x1, y1] from image 1 corresponds
            to [x2, y2] from image 2.

        Outputs:
            H: Homography matrix.
    '''
    fp = ones((len(plist), 3))
    tp = ones((len(plist), 3))

    for idx in range(len(plist)):
        fp[idx, 0] = plist[idx][0][0]
        fp[idx, 1] = plist[idx][0][1]

        tp[idx, 0] = plist[idx][1][0]
        tp[idx, 1] = plist[idx][1][1]

    H = Haffine_from_points(fp.T, tp.T)

    return H

def ransac(im1, im2, points_list, iters = 10 , error = 10, good_model_num = 5):
    '''
        This function uses RANSAC algorithm to estimate the
        shift and rotation between the two given images
    '''

    if ndim(im1) == 2:
        rows,cols = im1.shape
    else:
        rows, cols, _ = im1.shape

    model_error = 255
    model_H = None

    for i in range(iters):
        consensus_set = []
        points_list_temp = copy(points_list).tolist()
        # Randomly select 3 points
        for j in range(3):
            temp = choice(points_list_temp)
            consensus_set.append(temp)
            points_list_temp.remove(temp)

        # Calculate the homography matrix from the 3 points

        fp0 = []
        fp1 = []
        fp2 = []

        tp0 = []
        tp1 = []
        tp2 = []
        for line in consensus_set:

            fp0.append(line[0][0])
            fp1.append(line[0][1])
            fp2.append(1)

            tp0.append(line[1][0])
            tp1.append(line[1][1])
            tp2.append(1)

        fp = array([fp0, fp1, fp2])
        tp = array([tp0, tp1, tp2])

        H = Haffine_from_points(fp, tp)

        # Transform the second image
        # imtemp = transform_im(im2, [-xshift, -yshift], -theta)
        # Check if the other points fit this model

        for p in points_list_temp:
            x1, y1 = p[0]
            x2, y2 = p[1]

            A = array([x1, y1, 1]).reshape(3,1)
            B = array([x2, y2, 1]).reshape(3,1)

            out = B - dot(H, A)
            dist_err = hypot(out[0][0], out[1][0])
            if dist_err < error:
                consensus_set.append(p)


        # Check how well is our speculated model
        if len(consensus_set) >= good_model_num:
            dists = []
            for p in consensus_set:
                x0, y0 = p[0]
                x1, y1 = p[1]

                A = array([x0, y0, 1]).reshape(3,1)
                B = array([x1, y1, 1]).reshape(3,1)

                out = B - dot(H, A)
                dist_err = hypot(out[0][0], out[1][0])
                dists.append(dist_err)
            if (max(dists) < error) and (max(dists) < model_error):
                model_error = max(dists)
                model_H = H

    return model_H

if __name__ == "__main__":
    try:
        os.mkdir("temp")
    except OSError:
        pass

    try:
        # Load images from command prompt
        im1 = Image.open(sys.argv[1])
        im2 = Image.open(sys.argv[2])
    except IndexError:
        print('Usage: python ransac.py image1 image2')
        sys.exit()
    im1.convert('L').save('temp/1.pgm')
    im2.convert('L').save('temp/2.pgm')
    im1 = asarray(im1)
    im2 = asarray(im2)
    process_image('temp/1.pgm', 'temp/1.key')
    process_image('temp/2.pgm', 'temp/2.key')
    key1 = read_features_from_file('temp/1.key')
    key2 = read_features_from_file('temp/2.key')
    score = match(key1[1], key2[1])
    plist = get_points(key1[0], key2[0], score)
    plot_matches(im1,im2,key1[0],key2[0],score)
    
    # Compare ransac and simple homography matrix
    out_ransac = ransac(im1, im2, plist)
    out_simple = get_homography(plist)

    H_ransac = inv(out_ransac)
    H_simple = inv(out_simple)

    im_ransac = affine_transform2(im1,
                                  H_ransac[:2, :2],
                                  [H_ransac[0][2], H_ransac[1][2]])

    im_simple = affine_transform2(im1,
                                  H_simple[:2, :2],
                                  [H_simple[0][2], H_simple[1][2]])

    Image.fromarray(im2).show()
    Image.fromarray(im_ransac).show()
    Image.fromarray(im_simple).show()
