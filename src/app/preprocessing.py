import math
from PIL import ImageOps
from scipy import ndimage
import numpy as np
import cv2

# Adjust image colors (black and white)
def adjust_colors(img):
    img = img.convert("L")

    return img

# Centering the image
def centering_img(img):
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)
    
    return img

# Method to resize image
def resize_image(img):
    img = np.array(img)

    img = centering_img(img)

    rows, cols = img.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        img = cv2.resize(img, (cols, rows))

    cols_padding = (
        int(math.ceil((28 - cols) / 2.0)),
        int(math.floor((28 - cols) / 2.0)),
    )
    rows_padding = (
        int(math.ceil((28 - rows) / 2.0)),
        int(math.floor((28 - rows) / 2.0)),
    )
    img = np.lib.pad(img, (rows_padding, cols_padding), "constant")

    return img

def binary_image(image, threshold=128):
    binary_array = np.array(image.point(lambda x: 0 if x < threshold else 255), dtype=np.uint8)
    binary_array[binary_array < 128] = 0
    binary_array[binary_array >= 128] = 1
    return binary_array * 255

# Method to find the center of mass of the image
def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty

# Method shift an image
def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted

# Method to shift an image to center its center of mass
def adjust_center(img):
    shiftx, shifty = get_best_shift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted

    return img