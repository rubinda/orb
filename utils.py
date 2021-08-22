#!/usr/bin/env python3
#
# utility methods for FAST & BRIEF
# @author David Rubin
# @license MIT
import cv2
import numpy as np


def rotation_matrix(theta):
    """
    Construct a rotation matrix from the given theta
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]);


def smooth_gauss(image, variance=2, kernel_size=(9, 9)):
    """
    Smooth an image with a Gaussian filter. Default parameters correspond the article for the BRIEF descriptor.

    :param image: the image to be smoothed
    :param variance: the variance of the filter (default=2)
    :param kernel_size: the size of the filter kernel as a tuple (default=9x9)
    :return: the smoothed image
    """
    return cv2.GaussianBlur(image, kernel_size, variance)


def get_raw_img(image_name):
    """
    Return an array with image pixels (can be colored)
    :return: colored image as numpy array
    """
    # IMREAD_COLOR ignores transparency (!)
    return cv2.imread(image_name, cv2.IMREAD_COLOR)


def get_gray_img(image_name):
    """
    Open the self.image_name and return the grayscale values
    :return: array with grayscale image values
    """
    raw_image = get_raw_img(image_name)
    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    return gray_image


def get_pairs(feature_size=256, window_size=31):
    """
    Return a list of feature_size point (x,y) pairs

    :param feature_size:    the descriptor size to be used in BRIEF
    :param window_size:     the patch (window) size to be used in BRIEF
    """
    # Approach proposed by the professor
    # std = 1 / 5 * window_size
    # point_pairs2 = np.int32(np.random.randn(4, feature_size) * std)

    # Generate random point pairs
    # Using the approach G II: Gaussian(0, 1/25 * window_size^2)
    std = 0
    dev = 1 / 25 * (window_size * window_size)
    point_pairs = np.int32(np.random.normal(std, dev, (4, feature_size)))
    # Make sure the points are inside the window (patch)
    half_window_size = window_size // 2 -1
    brief_points = np.maximum(-half_window_size, np.minimum(point_pairs, half_window_size))
    return brief_points


def hamming_distance(a, b):
    """
    Calculate the Hamming distance between 2 binary arrays

    :param a:   numpy binary array 1 (0s and 1s)
    :param b:   numpy binary array 2 (0s and 1s)
    """
    return np.count_nonzero(a != b)