import numpy as np
import cv2

# DEFINING_POINTS_PIXELS = np.array([[91, 163],
#                                    [241, 163],
#                                    [98, 266],
#                                    [322, 265]])

DEFINING_POINTS_PIXELS = np.array([[91, 155],
                                   [241, 155],
                                   [98, 258],
                                   [322, 257]])

DEFINING_POINTS_CM = np.array([[0, 975],
                               [290, 975],
                               [000, -110],
                               [290, -110]])

HOMOGRAPHY, _ = cv2.findHomography(DEFINING_POINTS_PIXELS, DEFINING_POINTS_CM)

def get_coords(x, y):
    a = np.array([[x, y]], dtype='float32')
    a = np.array([a])
    return np.squeeze(cv2.perspectiveTransform(a, HOMOGRAPHY))
