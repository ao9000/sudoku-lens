import numpy as np
import cv2
import math
from imutils import contours


def extract_grid(image):
    # Reduce noise
    # Convert image to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    # Contour detection, assuming grid is the biggest contour in the image
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort from descending order
    cnt = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]

    # Use contours to transform grid into "Top down view/bird's eye view
    # Find perimeter
    peri = cv2.arcLength(cnt, True)
    # Find corners
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    # Identify corners
    corners = [(corner[0][0], corner[0][1]) for corner in approx]
    top_left, bottom_left, bottom_right, top_right = corners[0], corners[1], corners[2], corners[3]

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

    map_matrix = cv2.getPerspectiveTransform(np.array((top_left, top_right, bottom_right, bottom_left), dtype="float32"), new_dimensions)

    # Apply perspective wrap using provided matrix
    grid = cv2.warpPerspective(image, map_matrix, (new_width, new_height))

    return grid


def filter_non_square_contours(cnts):
    # Define temp list
    square_indexes = []

    # Contour Approximation to find squares
    for cnt_index, cnt in enumerate(cnts, start=0):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Check if square
        if len(approx) == 4:
            # Get width & height of contour
            _, _, width, height = cv2.boundingRect(approx)

            # Compute aspect ratio
            aspect_ratio = width / height

            # Square will have aspect ratio of around 1
            if 0.85 <= aspect_ratio <= 1.15:
                # Append into list
                square_indexes.append(cnt_index)

    # Filter list to only contain square contours
    cnts = [cnt for cnt_index, cnt in enumerate(cnts) if cnt_index in square_indexes]

    return cnts


def sort_grid_contours(cnts):
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
