#!/usr/bin/env python3
#
# @author David Rubin
# @license MIT
import cv2
import numpy as np
from fast import FAST
from brief import BRIEF
from pathlib import Path
from utils import get_pairs, hamming_distance
from matplotlib import pyplot as plt


class ORB:
    def __init__(self, img1, img2, thr):
        """
        Create an ORB (-ish) comparator between two images.

        :param img1:    image 1
        :param img2:    image 2
        :param thr:     FAST threshold
        """
        self.image1 = img1
        self.image2 = img2
        self.keypoints1 = []
        self.keypoints2 = []
        self.threshold = thr

    def match_keypoints(self):
        """
        Will find key points on both images and try to match them using Hamming distance
        """
        # Instantiate FAST
        fast1 = FAST(self.threshold, Path(self.image1).absolute())
        fast2 = FAST(self.threshold, Path(self.image2).absolute())

        # Instantiate BRIEF
        descriptor_size = 256
        patch_size = 31
        use_steered = True
        pairs = get_pairs(feature_size=descriptor_size, window_size=patch_size)
        brief1 = BRIEF(self.image1, pairs)
        brief2 = BRIEF(self.image2, pairs)

        # Find key points and their orientations
        min_edge = patch_size // 2 + 1
        max_edge = np.add(brief1.image.shape, (-min_edge, -min_edge))
        kp1 = fast1.find_corners(nonmax=True)
        keypoints1, orientations1 = self.find_orientations(min_edge, max_edge, kp1, brief1)
        self.keypoints1 = np.array([list(k) for k in kp1.keys()])

        kp2 = fast2.find_corners(nonmax=True)
        max_edge = np.add(brief2.image.shape, (-min_edge, -min_edge))
        keypoints2, orientations2 = self.find_orientations(min_edge, max_edge, kp2, brief2)
        self.keypoints2 = np.array([list(k) for k in kp2.keys()])

        # Find descriptors using steered BRIEF
        descriptors1 = []
        for point, theta in zip(keypoints1, orientations1):
            d = brief1(point, steered=use_steered, theta=theta)
            descriptors1.append((point, d))
        descriptors2 = []
        for point, theta in zip(keypoints2, orientations2):
            d = brief2(point, steered=use_steered, theta=theta)
            descriptors2.append((point, d))

        # Match using Hamming distance
        matched_points = []
        max_similar_distance = 35
        for index1, (point, descriptor1) in enumerate(descriptors1):
            distances = [hamming_distance(descriptor1, descriptor2) for _, descriptor2 in descriptors2]
            if np.min(distances) < max_similar_distance:
                index2 = np.argmin(distances)
                # nearest_point = descriptors2[np.argmin(distances)][0]
                matched_points.append([index1, index2])
        return np.array(matched_points)

    def find_orientations(self, min_edge, max_edge, keypoints, brief):
        """
        Return key points that have valid patches (inside bounds) and their orientations
        """
        nonedge_keypoints = []
        orientations = []
        for point in keypoints:
            if brief.invalid_patch_bounds(point):
                # Skip this point as it does not have a valid patch area
                continue
            #o = brief.corner_orientation(point)
            o, _ = brief.calculate_orientation(point)
            # print(f'Davids: {o2}\nOther: {o}')
            # assert abs(o - o2) < 10e-2
            nonedge_keypoints.append(point)
            orientations.append(o)
        return nonedge_keypoints, orientations


if __name__ == '__main__':
    from skimage.feature import plot_matches

    img1 = 'images/chess_queen.jpg'
    img2 = 'images/chess.jpg'

    # Instantiate ORB
    fast_threshold = 25
    orb = ORB(img1, img2, fast_threshold)

    matches = orb.match_keypoints()
    #
    # Plot the matched points
    im1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)

    plot_matches(ax, im1, im2, orb.keypoints1, orb.keypoints2, matches)
    plt.axis('off')
    plt.show()
