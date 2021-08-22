#!/usr/bin/env python3
#
# @author David Rubin
# @license MIT
import cv2
import numpy as np
from typing import Tuple
from pathlib import Path
from utils import get_gray_img, get_raw_img, smooth_gauss


class FAST:
    """
    A FAST detector (Rosten 2006) for keypoint detection.
    """

    def __init__(self, thr, img, levels=0, smooth=False):
        self.results_dir = Path('./results')
        self.results_dir.mkdir(exist_ok=True)
        self.threshold = thr
        self.image_name = str(img)
        self.image = get_gray_img(self.image_name)
        if smooth:
            self.image = smooth_gauss(self.image)
        # How many pixels must be brighter or darker than the center
        self.n = 12
        # Number of pyramid levels to use
        self.image_pyramid = [np.copy(self.image)]
        for level in range(levels):
            self.image_pyramid.append(cv2.pyrDown(self.image_pyramid[level]))
        self.high_speed_candidates = {}     # Keep track of high speed candidates (needed for report on task)

    def name_file(self, suffix):
        """
        Returns a file name for a new image based on self.image_name and whatever suffix is provided
        Uses self.results_dir as directory
        """
        og_file = Path(self.image_name)
        return f'{self.results_dir}/{og_file.stem}_{suffix}{og_file.suffix}'

    def paint_corners(self, image, corners, suffix=None):
        """
        Reads the image and marks the corners as green pixels (lawngreen   #7CFC00     rgb(124,252,0)
        Also creates a new file called <image_name>_corners.<extension> where you can view the image.
        Also note that OpenCV uses BGR instead of RGB

        :param image    image we should draw corner on
        :param corners: the corner pixels we would like to paint (dict with tuples as keys)
        :param suffix:  custom suffix to use for filename (default='')
        :return: filename of the corner painted image
        """
        new_filename = self.name_file(f'corners{suffix if suffix is not None else ""}')
        img = np.copy(image)    # Don't modify original
        for corner in corners.keys():
            img[corner][0] = 0  # The blue value
            img[corner][1] = 252  # The green value
            img[corner][2] = 124  # The red value
        cv2.imwrite(new_filename, img)
        return new_filename

    def circle_corners(self, image, corners, r=4, suffix=None):
        """
        Draw circles with corners as center points on image

        :param image:   image to paint on
        :param corners: dict with corners, used as center of circle
        :param r:       radius (in pixels) of circle
        :param suffix:  custom suffix to use for filename (default='')
        :return: the filename of the new image
        """
        new_filename = self.name_file(f'cornersBig{suffix if suffix is not None else ""}')
        img = image.copy()
        r_half = r//2
        for x, y in corners.keys():
            cv2.circle(img, (int(y), int(x)), r, (0, 255, 0), 1)
        cv2.imwrite(new_filename, img)

    def nonmax_supression(self, image, corners: dict):
        """
        Performs non-maximal suppression on the corners.
        If two corners are adjacent, then discard the one with the lower score, otherwise do nothing.

        It paints the corners into an image (scores represent values) and removes the lower valued neighbours.
        :param image:   image to use as base for corner score mask
        :param corners: dict with points (tuples) as keys and corner scores as values
        :return: dict with lesser corners (some are suppressed)
        """
        # Paint an image with corner scores being the only values >0
        sc = np.zeros(image.shape)
        for corner, score in corners.items():
            sc[corner] = score

        nonmax_corners = {}
        # Check the neighbours of every corner pixel, if all have a lower score append the center one
        for corner, score in corners.items():
            x = corner[0]
            y = corner[1]
            if score >= sc[x - 1][y - 1] and score >= sc[x - 1][y] and score >= sc[x - 1][y + 1] \
                    and score >= sc[x][y - 1] and score >= sc[x][y + 1] and score >= sc[x + 1][y - 1] \
                    and score >= sc[x + 1][y] and score >= sc[x + 1][y + 1]:
                # The center score is bigger than all neighbouring values
                nonmax_corners[corner] = score
        return nonmax_corners

    def high_speed_test(self, image, pixel: Tuple[int, int]):
        """
        Implements a high speed test whether a given pixel is a corner (circle of 16 pixels)
        TODO: might be faster to check only north/south first and return false if both are not brighter or darker

        :param image:   image to use
        :param pixel: the center pixel we are testing (tuple)
        :return: true/false if corner and true/false if brighter
        """
        center = image[pixel[0], pixel[1]]
        neighbours = np.array([image[pixel[0], pixel[1] - 3], image[pixel[0] + 3, pixel[1]],
                               image[pixel[0], pixel[1] + 3], image[pixel[0] - 3, pixel[1]]])

        # The threshold for brightness and darkness
        thr_bright = center + self.threshold
        thr_dark = center - self.threshold
        # A value higher or equal to 3 suggests that we might have a corner
        if sum(neighbours > thr_bright) >= 3:
            return True, True
        if sum(neighbours < thr_dark) >= 3:
            return True, False
        # Not a candidate
        return False, None

    def bresenham_circle(self, image, pixel: Tuple[int, int]):
        """
        Returns pixels forming a Bresenham circle (N=16) using given pixel coordinates as

        :param image:   image to take pixels from
        :param pixel: tuple representing circle center
        """
        x, y = pixel
        return [
            image[x, y - 3],
            image[x + 1, y - 3],
            image[x + 2, y - 2],
            image[x + 3, y - 1],
            image[x + 3, y],
            image[x + 3, y + 1],
            image[x + 2, y + 2],
            image[x + 1, y + 3],
            image[x, y + 3],
            image[x - 1, y + 3],
            image[x - 2, y + 2],
            image[x - 3, y + 1],
            image[x - 3, y],
            image[x - 3, y - 1],
            image[x - 2, y - 2],
            image[x - 1, y - 3],
        ]

    def corner_test(self, image, pixel: Tuple[int, int], high_speed_brighter):
        """
        Perform a corner test on a given center pixel.

        :param image:   image to use for pixel values
        :param pixel: tuple containing pixel coordinates (x, y)
        :param high_speed_brighter: result of the high speed test (3 brighter or darker pixels were found)
        :return: False if not a corner, True if a corner is detected and the corner score value
        """
        # Check if pixel coordinates are valid (you can form a 16 point circle around them)
        max_edge = np.add(pixel, (3, 3))
        min_edge = np.add(pixel, (-3, -3))
        if (max_edge > image.shape).any() or (min_edge < (0, 0)).any():
            raise ValueError('Given pixel {} is out of bounds for image shape {}.'.format(pixel, image.shape))

        center_value = int(image[pixel])

        # Values on the circle around the center pixel
        neighbour_values = self.bresenham_circle(image, pixel)

        # Make sure at least 12 contiguous pixels are all brighter/darker than the threshold
        max_contiguous = 0
        curr_contiguous = 0

        # Compute the corner score only for the possibility suggested by the high speed test
        if high_speed_brighter:
            # circle_pixel > center_value + threshold => circle_pixel - center_value - threshold > 0
            pixels_score = [p - center_value - self.threshold for p in neighbour_values]
        else:
            # circle_pixel < center_value - threshold => center_value - threshold - circle_pixel > 0
            pixels_score = [center_value - p - self.threshold for p in neighbour_values]
        for pixel in pixels_score * 2:
            # Iterate over the list twice (we are checking a circle, perhaps pixels are contiguously brighter over our
            # starting/ending portion)
            curr_contiguous = curr_contiguous + 1 if pixel > 0 else 0
            if curr_contiguous > max_contiguous:
                max_contiguous = curr_contiguous
        pixels_score = np.array(pixels_score)
        # Return tuple of are there N contiguous pixels that satisfy condition, score of those pixels
        return max_contiguous >= self.n, np.sum(pixels_score[pixels_score > 0])

    def _find_corners(self, image, nonmax=False):
        """
        Looks for corners on the image passed

        :param nonmax: should the neighbouring corners be suppressed using non-maximal suppression
        :return: map of corners, where keys are coordinates and value is corner score
        """
        corners = {}
        self.high_speed_candidates = {}
        print(f'Running corner detection on {self.image_name} {image.shape}')
        for i in range(3, image.shape[0] - 3):
            for j in range(3, image.shape[1] - 3):
                current_pixel = (i, j)
                # If the high speed test confirms the pixel, try it with the normal variant
                is_candidate, is_brighter = self.high_speed_test(image, current_pixel)
                if is_candidate:
                    # Test the more promising pixel around the whole circle
                    is_corner, score = self.corner_test(image, current_pixel, is_brighter)
                    if is_corner:
                        # Pixel seems to be a corner, append it
                        corners[current_pixel] = score
                    self.high_speed_candidates[current_pixel] = True    # Keep track of all high speed candidates
        if nonmax:
            return self.nonmax_supression(image, corners)
        return corners

    def find_corners(self, nonmax=False):
        """
        Looks for corners on the original image passed

        :param nonmax: should the neighbouring corners be suppressed using non-maximal suppression
        :return: map of corners, where keys are coordinates and value is corner score
        """
        return self._find_corners(self.image, nonmax)

    def find_scaled_corners(self, nonmax=False):
        """
        Looks for corners on all of the scaled images (defined by levels on init)

        :param nonmax: should the neighbouring corners be suppressed using non-maximal suppression
        :return: list of corner maps as described in self._find_corners
        """
        scaled_keypoints = []
        for image in self.image_pyramid:
            corners = self._find_corners(image, nonmax)
            scaled_keypoints.append(corners)
        return scaled_keypoints


if __name__ == '__main__':
    image_name = 'images/testna_slika.png'
    img = get_raw_img(image_name)
    threshold = 15      # How much pixels must differ for a darker/brighter assumption
    nonmax = True

    fast = FAST(threshold, Path(image_name).absolute(), levels=2)
    # fast_points is a dict with pixel locations as keys and corner score as value
    fast_points = fast.find_corners(nonmax=nonmax)
    fast.paint_corners(img, fast_points, suffix='_nmax')
    # fast.paint_corners(img, fast.high_speed_candidates, suffix='_high-speed')
    n_corners = len(fast_points)
    # n_candidates = len(fast.high_speed_candidates)
    print(f'FAST results:')
    print(f' {"image:":13} "{image_name}"')
    print(f' {"threshold:":13} "{threshold}"')
    print(f' {"nonmax?:":13} "{nonmax}"')
    # print(f' {"candidates:":13} {n_candidates:12}  (high speed test)')
    print(f' {"corners:":13} {n_corners:12}  (16 pixels test)')
    # print(f' {"difference:":13} {n_candidates-n_corners:12}  (candidates - corners)')
    # print(f' {"ratio:":13} {n_corners/n_candidates:12.3f}  (corners/candidates)')

    # sc_keypoints = fast.find_scaled_corners(nonmax=True)
    # scale_img = img.copy()
    # r = 4   # Base radius
    # for i, keypoints in enumerate(sc_keypoints):
    #     fast.circle_corners(scale_img, keypoints, r, f'_L{i}')
    #     scale_img = cv2.pyrDown(scale_img)
