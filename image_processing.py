"""
    Contains the helper function for extracting the sudoku grid from the given raw image
"""


import numpy as np
import cv2
import math
from imutils import contours


def get_grid_dimensions(image):
    """
    Tries to locate the grid dimension from a raw image

    Process:
    1. Converts the given image into greyscale
    2. Apply gaussian blur to smooth the image
    3. Apply thresholding to obtain a inverted binary image (Lines will be white while background will be black)
    4. Apply contour detection and sort the contours from largest to smallest
    5. Loop and filter the contour. Finding the largest and square contour (Assume the largest square contour will be the grid)
    6. Returns the grid dimensions

    :param image: type: numpy.ndarray
    The raw sudoku image in color format

    :return: type: tuple if grid is found, else None
    Returns the grid coordinates in top left, top right, bottom right, bottom left order. If no grid is found, returns None
    """
    # Reduce noise
    # Convert image to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    # Contour detection, assuming grid is the biggest contour in the image
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort from descending order & loop
    for cnt in sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True):
        # Use contours to transform grid into "Top down view/bird's eye view
        # Find perimeter
        peri = cv2.arcLength(cnt, True)
        # Find corners
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Get area of contour
        cnt_area = cv2.contourArea(cnt)

        # Filter only square/rect for grid
        # Also filter for the grid to be at least 50% of the image size
        if cnt_area >= 0.3 * (image.shape[0] * image.shape[1]):
            if len(approx) == 4:
                # Unpack & Identify corners
                corners = sorted([(corner[0][0], corner[0][1]) for corner in approx], key=lambda x: x[0], reverse=False)
                top_corners, bottom_corners = corners[:-2], corners[2:]
                top_left, bottom_left = sorted(top_corners, key=lambda x: x[1], reverse=False)
                top_right, bottom_right = sorted(bottom_corners, key=lambda x: x[1], reverse=False)

                # Return detected grid dimensions
                return top_left, top_right, bottom_right, bottom_left

    # Unable to find grid
    return None


def transform_grid(image, grid_coordinates):
    """
    Crops the detected grid and transforms it to reduce noise.

    Process:
    1. Calculates the Euclidean distance of the grid to have a perfect cropping without losing any information
    2. Performs perspective wrap to correct images with skewing

    :param image: type: numpy.ndarray
    The raw sudoku image in color format

    :param grid_coordinates: type: tuple
    Grid coordinates in top left, top right, bottom right, bottom left order

    :return: type: numpy.ndarray
    Transformed grid
    """

    # Unpack
    top_left, top_right, bottom_right, bottom_left = grid_coordinates

    # Calculate the Euclidean distance for top & bottom width
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    # Select the max of both
    new_width = int(max(top_width, bottom_width))

    # Calculate the Euclidean distance for left & right height
    left_height = np.sqrt(((bottom_left[0] - top_left[0]) ** 2) + ((bottom_left[1] - top_left[1]) ** 2))
    right_height = np.sqrt(((bottom_right[0] - top_right[0]) ** 2) + ((bottom_right[1] - top_right[1]) ** 2))
    # Select the max of both
    new_height = int(max(left_height, right_height))

    # Construct the new image frame dimensions
    # [0, 0] - Top left
    # [new_width - 1, 0] - Top right
    # [new_width - 1, new_height - 1] - Bottom right
    # [0, new_height - 1] - Bottom left
    new_dimensions = np.array([
        [0, 0],
        [new_width - 1, 0],
        [new_width - 1, new_height - 1],
        [0, new_height - 1]],
        dtype="float32")

    map_matrix = cv2.getPerspectiveTransform(
        np.array((top_left, top_right, bottom_right, bottom_left), dtype="float32"), new_dimensions)

    # Apply perspective wrap using provided matrix
    grid = cv2.warpPerspective(image, map_matrix, (new_width, new_height))

    return grid


def filter_non_square_contours(cnts):
    """
    Filters the contour list to remove contours that are not square shaped

    :param cnts: type: list
    List of contours detected

    :return: type: list
    Filtered list of contour
    """
    # Define temp list
    square_indexes = []

    # Contour Approximation to find squares
    for cnt_index, cnt in enumerate(cnts, start=0):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Check if square
        if len(approx) == 4:
            # # Get width & height of contour
            # _, _, width, height = cv2.boundingRect(approx)

            # Change to rotatedRect
            rect = cv2.minAreaRect(approx)
            width, height = rect[1]

            # Compute aspect ratio
            aspect_ratio = width / height

            # Square will have aspect ratio of around 1
            tol = 0.15  # Tolerance for aspect ratio
            if (1.0 - tol) <= aspect_ratio <= (1.0 + tol):
                # Append into list
                square_indexes.append(cnt_index)

    # Filter list to only contain square contours
    cnts = [cnt for cnt_index, cnt in enumerate(cnts) if cnt_index in square_indexes]

    return cnts


def sort_grid_contours(cnts):
    """
    Sorts the list of contours based on their location on the image (Top to bottom then left to right)
    Then constructs the 9x9 nested list simulating the sudoku board and appends appends the contours into the list

    :param cnts: type: list
    List of contours detected

    :return: type: list
    9x9 nested list containing contour information for each cell
    """

    grid_contours = [[] for _ in range(0, 9)]

    # Sort contours (From top to bottom and left to right)
    cnts, _ = contours.sort_contours(cnts, method="top-to-bottom")

    # Extract every row
    for row_index, row in enumerate(range(0, 81, 9), start=0):
        row_cnts, _ = contours.sort_contours(cnts[row:row + 9], method="left-to-right")
        for box_cnt in row_cnts:
            # Append
            grid_contours[row_index].append(box_cnt)

    return grid_contours


def reduce_noise(grid):
    """
    Prepare the grid for contour detection
    The goal is to reduce as much noise on the grid so that we are able to detect 81 cells perfectly

    Process:
    1. Convert input grid into greyscale
    2. Perform contour detection to obtain binary image (White lines, black background)
    3. Detect contour to get the bounding boxes for all the grid cells
    4. Filter the contour for small contours
    5. Cover up the digits on the cells to reduce noise
    6. Perform closing to strengthen the cell boarders for easier contour detection
    7. Invert the image back to normal

    :param grid: type: numpy.ndarray
    The image of the cropped and transformed grid

    :return: type: type: numpy.ndarray
    The image of the cropped and transformed grid after noise reduction
    """

    # Convert image to greyscale
    gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding, invert resulting image for morphological operations
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 7)

    # Contour detection, detect any noises such as Dots/numbers
    # Best detecting white objects on a black background
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find only small contours
    area_thresh = math.prod([num * 1/15 for num in grid.shape[:2]])
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) < area_thresh]

    # Draw contours over numbers on the grid
    cv2.drawContours(thresh, cnts, -1, (0, 0, 0), -1)

    # Define vertical kernel to fix lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    # Define horizontal kernel to fix lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

    # Closing to fix noise in grid lines
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=7)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=7)

    # Invert the image back to normal
    thresh = cv2.bitwise_not(thresh)

    return thresh


def get_cells_from_9_main_cells(cnts):
    # New cnts to store cleaned cnts
    new_cnts = []

    # Main cells extracted, split main cells to 81 cells
    # Get individual main cell height and width for splitting
    for cnt in cnts:
        # Get bounding box for each contour
        x, y, width, height = cv2.boundingRect(cnt)

        # Calculate individual cell height and width
        cell_height = height // 3
        cell_width = width // 3

        # Split contour into 9 cells
        for row_num in range(1, 4):
            for col_num in range(1, 4):
                new_cnt = np.array([[(x + (cell_width * (col_num - 1)), y + (cell_height * (row_num - 1))),
                                     (x + (cell_width * col_num), y + (cell_height * (row_num - 1))),
                                     (x + (cell_width * col_num), y + (cell_height * row_num)),
                                     (x + (cell_width * (col_num - 1)), y + (cell_height * row_num))]], dtype=np.int32)
                new_cnts.append(new_cnt)

    return new_cnts


def is_blur(image, thresh=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lv = cv2.Laplacian(gray, cv2.CV_64F).var()

    if lv < thresh:
        return True, lv
    else:
        return False, lv
