#!/usr/bin/env python3
#
# @author David Rubin
# @license MIT
import numpy as np
from fast import FAST
from pathlib import Path
from utils import get_gray_img, smooth_gauss, rotation_matrix, get_raw_img, get_pairs


class BRIEF:
    """
    Binary Robust Independent Elementary Features - BRIEF implementation.

    With the help of Calonder et. al. BRIEF: Binary Robust Independent Elementary Features
    """

    def __init__(self, image_name, pairs, window_size=31, feature_size=256):
        """
        Create a new BRIEF descriptor.

        THe default values are derived from the article.

        :param image_name: the path to an image
        :param pairs:  2 lists of points that should match in length (see get_pairs(...))
        :param window_size: size of the patch (default is 31)
        :param feature_size: size of the binary feature vector in bits (default is 256)
        """
        self.window_size = window_size
        self.feature_size = feature_size
        self.image = get_gray_img(image_name)
        self.pairs = pairs

        # Smooth the image with a Gaussian kernel (removes sensitivity to high-frequency noise)
        self.image_smooth = smooth_gauss(self.image)

    def __call__(self, keypoint, steered=False, theta=0):
        """
        Calculate the BRIEF feature vector for the given key point.

        :param keypoints: the detected keypoints (see FAST)
        :param steered:  if the orientation should be provided
        :param theta: orientation of the keypoint
        :return: BRIEF.feature_size sized binary vector
        """
        # Check if the patch around the keypoint overflows the image
        img_height, img_width = self.image.shape
        half_window_size = self.window_size // 2
        distance = half_window_size    # Masking with safe distance from border
        if steered:
            distance *= 1.5     # Add some more distance in case of steered version

        if keypoint[0] < distance or keypoint[1] < distance \
                or keypoint[1] >= (img_width - distance) or keypoint[0] >= (img_height - distance):
            # Invalid keypoint (the patch around it exceeds the image boundaries
            return None

        # Move the BRIEF points onto the FAST keypoint
        keypoint = np.array(keypoint).reshape(2, 1)
        pairs1 = self.pairs[::2, :]
        pairs2 = self.pairs[1::2, :]
        # When using the steered version multiply the points with a rotation matrix (to produce S_theta)
        # Note that both steered=True and theta=<value> must be given for proper usage
        if steered:
            pairs1 = self.steer_pairs(self.pairs[::2, :], theta)
            pairs2 = self.steer_pairs(self.pairs[1::2, :], theta)

        # Compare the keypoint with the values in the patch
        fb_points1 = keypoint + pairs1
        fb_points2 = keypoint + pairs2

        descriptor = self.image_smooth[fb_points1[0, :], fb_points1[1, :]] > \
            self.image_smooth[fb_points2[0, :], fb_points2[1, :]]
        return np.array(descriptor, dtype=np.int8)

    def steer_pairs(self, pairs, theta):
        """
        Rotates the point pairs and returns new arrays for them
        The idea is that
        x' = x * cos(theta) - y * sin(theta)
        y' = x * sin(theta) + y * cos(theta)

        :param pairs: 2D list of point pairs [0, :] is x, [1, :] is y
        :param theta: orientation defined user parameter
        :returns new lists containing rotated points pairs
        """
        pairs_x = (pairs[0, :] * np.cos(theta)) - (pairs[1, :] * np.sin(theta))
        pairs_y = (pairs[0, :] * np.sin(theta)) + (pairs[1, :] * np.cos(theta))

        return [np.rint(pairs_x).astype(int), np.rint(pairs_y).astype(int)]

    def patch_moment(self, p, q, center):
        """
        Return the moment of a patch around the given keypoint.

        If the patch is a circle with diameter of 2r,
        then X and Y values range from -r to r.

        :param p: the moment parameter for x
        :param q: the moment parameter for y
        :param center: the center of the patch
        :return: the moment
        """
        r = int((self.window_size-1) / 2)
        # Check if the center is at least r away from the image edges
        max_edge = np.add(center, (r + 1, r + 1))
        min_edge = np.add(center, (-r, -r))
        if (max_edge > self.image.shape).any() or (min_edge < (0, 0)).any():
            print(f'Min edge is {min_edge}, max edge is {max_edge}')
            raise ValueError(f'Given center for the patch moment {center} is out of bounds for image shape {self.image.shape}.')

        # "To improve the rotation invariance of this measure we
        # make sure that moments are computed with x and y remaining
        # within a circular region of radius r. We empirically
        # choose r to be the patch size, so that that x and y run from
        # [−r, r]." as per ORB article
        # fixme: uses fixed window size of 31
        circle_mask = np.zeros((31, 31), dtype=np.int32)
        # Represents the number of pixels from center that create a circle
        circle_dists = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
        for i in range(-15, 16):
            for j in range(-circle_dists[abs(i)], circle_dists[abs(i)] + 1):
                circle_mask[15 + j, 15 + i] = 1

        patch = self.image_smooth[center[0] - r: center[0] + r + 1, center[1] - r: center[1] + r + 1]
        moment_pq = 0
        for y in range(patch.shape[0]):
            for x in range(patch.shape[1]):
                if circle_mask[x, y]:
                    moment_pq += (x-r) ** p * (y-r) ** q * patch[x][y]
        return moment_pq

    def invalid_patch_bounds(self, keypoint):
        """
        Returns true if keypoint is outside the patch for orientation calculation
        """
        min_edge = self.window_size // 2 + 1
        max_edge = np.add(self.image.shape, (-min_edge, -min_edge))
        return keypoint[0] < min_edge or keypoint[1] < min_edge or keypoint[0] > max_edge[0] or keypoint[1] > max_edge[1]

    def calculate_orientation(self, keypoint):
        """
        Calculate centroid and rotation of a given keypoint (orb_final.pdf, 3.2)

        :param keypoint: pixel within certain bounds (window_size) on image
        :return orientation
        """
        m_10 = self.patch_moment(1, 0, keypoint)
        m_01 = self.patch_moment(0, 1, keypoint)
        m_00 = self.patch_moment(0, 0, keypoint)
        centroid = (m_10 / m_00, m_01 / m_00)
        return np.arctan2(m_01, m_10), centroid

    def corner_orientation(self, keypoint):
        half_w = int((self.window_size - 1) / 2)
        OFAST_MASK = np.zeros((31, 31), dtype=np.int32)
        OFAST_UMAX = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
        for i in range(-15, 16):
            for j in range(-OFAST_UMAX[abs(i)], OFAST_UMAX[abs(i)] + 1):
                OFAST_MASK[15 + j, 15 + i] = 1

        img = np.pad(self.image, (half_w, half_w), mode='constant', constant_values=0)
        m01, m10 = 0, 0
        for r in range(self.window_size):
            for c in range(self.window_size):
                if OFAST_MASK[c, r]:
                    I = img[keypoint[0] + c, keypoint[1] + r]
                    m10 = m10 + I * (c - half_w)
                    m01 = m01 + I * (r - half_w)
        return np.arctan2(m01, m10)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    image_name = 'images/chess_queen5.jpg'
    img = get_raw_img(image_name)
    descriptor_size = 256
    window_size = 31
    threshold = 20

    fast = FAST(threshold, Path(image_name).absolute())
    # Keypoints is a dict with pixel locations as keys and corner score as values
    keypoints = fast.find_corners(nonmax=True)

    # Create point pairs for each keypoint (using a normal distribution)
    pairs = get_pairs(descriptor_size, window_size)
    # Initialize the BRIEF descriptor
    brief = BRIEF(image_name, pairs, window_size, descriptor_size)

    # Calculates a descriptor for each keypoint (call the BRIEF object)
    descriptors = []
    keypoint_origins = []
    keypoint_directions = []
    thetas = []
    for kp in keypoints.keys():
        # Returns orientation angle (theta from equation) and the direction to the intensity center
        if brief.invalid_patch_bounds(kp):
            continue

        orientation, center = brief.calculate_orientation(kp)
        # print(f'KP is {kp}')
        # print(f'  Center is {center}, orientation is {orientation}')
        keypoint_origins.append(list(kp))
        keypoint_directions.append(list(center))
        descriptors.append(brief(kp, steered=True, theta=orientation))
        thetas.append(orientation)
    #
    keypoint_origins = np.array(keypoint_origins)
    keypoint_directions = np.array(keypoint_directions)
    plt.imshow(img)
    kp_index = 0
    keypoint = np.array(keypoint_origins[kp_index]).reshape((2, 1))
    theta = thetas[kp_index]
    brief_pts1 = brief.steer_pairs(pairs[:2, :10], theta) + keypoint
    plt.scatter(keypoint[1], keypoint[0])
    plt.scatter(brief_pts1[1], brief_pts1[0], color='orange')
    plt.axis('off')
    plt.show()

    image_name = 'images/chess_queen_25.jpg'
    kp_index2 = 3
    img = get_raw_img(image_name)
    fast = FAST(threshold, Path(image_name).absolute())
    # Keypoints is a dict with pixel locations as keys and corner score as values
    keypoints = fast.find_corners(nonmax=True)
    brief = BRIEF(image_name, pairs, window_size, descriptor_size)

    keypoints2 = list(keypoints.keys())

    theta2, _ = brief.calculate_orientation(keypoints2[kp_index2])

    keypoint2 = np.array(list(keypoints2[kp_index2]), dtype=np.int).reshape((2, 1))
    print(keypoint2)
    brief_pts2 = brief.steer_pairs(pairs[:2, :10], theta2) + keypoint2
    # brief_pts2 = pairs[:2, :10] + keypoint2
    plt.imshow(img)
    plt.scatter(keypoint2[1], keypoint2[0])
    plt.scatter(brief_pts2[1], brief_pts2[0])
    plt.axis('off')
    plt.show()

    # plt.quiver(keypoint_origins[:, 1], keypoint_origins[:, 0], keypoint_directions[:, 0], keypoint_directions[:, 1],
    #            color='limegreen')


