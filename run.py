import numpy as np
from backtracking import backtracking
import cv2
import math
from imutils import contours

def create_board():
    # Create a blank Sudoku board
    board = np.zeros((9, 9), dtype=int)

    # Sample Board config
    board[0] = [0, 0, 0, 2, 6, 0, 7, 0, 1]
    board[1] = [6, 8, 0, 0, 7, 0, 0, 9, 0]
    board[2] = [1, 9, 0, 0, 0, 4, 5, 0, 0]
    board[3] = [8, 2, 0, 1, 0, 0, 0, 4, 0]
    board[4] = [0, 0, 4, 6, 0, 2, 9, 0, 0]
    board[5] = [0, 5, 0, 0, 0, 3, 0, 2, 8]
    board[6] = [0, 0, 9, 3, 0, 0, 0, 7, 4]
    board[7] = [0, 4, 0, 0, 5, 0, 0, 3, 6]
    board[8] = [7, 0, 3, 0, 1, 8, 0, 0, 0]

    return board


# def crop_grid(image):
#     # Get image width & height
#     height, width = image.shape[:2]
#
#     # Convert image to greyscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Image processing
#     # Gaussian blur
#     blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)
#
#     # Gaussian thresholding
#     thres = cv2.adaptiveThreshold(src=blur, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=0)
#
#     # Invert image
#     invert = cv2.bitwise_not(thres)
#
#     # Define kernel
#     kernel = np.ones((3, 3), np.uint8)
#     # Dilate image
#     dilate = cv2.dilate(invert, iterations=1, kernel=kernel)
#
#     # Contour detection, assuming grid is the biggest contour in the image
#     contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # Sort form descending order
#     contours = sorted([cv2.boundingRect(contour) for contour in contours], key=lambda x: x[2] * x[3], reverse=True)
#
#     # Assume second largest contour is the grid
#     x, y, w, h = contours[1]
#
#     # Crop image
#     grid = image[y:y+h+int((1/100)*height), x:x+w+int((1/100)*width)]
#
#     return grid
#
#
# def get_lines(grid):
#     # Convert image to greyscale
#     gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
#
#     # Image processing
#     # Gaussian blur
#     gray = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)
#
#     # Canny edge detection
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#
#     # Dilate then erode image, which means closing
#     # Define kernel
#     kernel = np.ones((3, 3), np.uint8)
#     closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#
#     # Hough transform line detection
#     lines = cv2.HoughLines(closing, 1, 1 * (np.pi / 180), 200)
#
#     return lines
#
#
# def filter_lines(lines):
#     # Filter results, assuming the grid is upright
#     # Filter all lines except for vertical 0/180 degrees & horizontal 90 degrees lines with threshold
#     theta_degree_filter_threshold = 5
#     vertical_lines = lines[
#         (abs(lines[:, 0, 1] * (180 / np.pi) - 0) <= theta_degree_filter_threshold) |
#         (abs(lines[:, 0, 1] * (180 / np.pi) - 180) <= theta_degree_filter_threshold)
#         ]
#
#     horizontal_lines = lines[
#         (abs(lines[:, 0, 1] * (180 / np.pi) - 90) <= theta_degree_filter_threshold)
#     ]
#
#     return vertical_lines, horizontal_lines
#
#
# def remove_duplicated_lines(lines, rho_threshold, theta_threshold):
#     # Merge/remove similar lines
#     similar_lines = []
#
#     # Outer loop
#     for outer_index in range(len(lines)):
#         if any(outer_index in set_lines for set_lines in similar_lines):
#             continue
#         outer_rho, outer_theta = lines[outer_index][0]
#         temp = [outer_index]
#
#         # Inner loop
#         for inner_index in range(len(lines)):
#             if inner_index == outer_index:
#                 continue
#             inner_rho, inner_theta = lines[inner_index][0]
#             if outer_rho-rho_threshold < inner_rho < outer_rho+rho_threshold and \
#                     outer_theta-theta_threshold < inner_theta < outer_theta+theta_threshold:
#                 temp.append(inner_index)
#
#         similar_lines.append(temp)
#
#     # Find median
#     new_lines = []
#     for set_lines in similar_lines:
#         temp = sorted([lines[index] for index in set_lines], key=lambda x: x[0][0])
#         new_lines.append(temp[math.floor((len(temp)-1)/2)])
#
#     return new_lines
#
#
# def find_intersection(line1, line2):
#     # See https://stackoverflow.com/a/383527
#     # Line equation: ρ = x * cosθ + y * sinθ
#     # Intersection of 2 lines, solve for x,y
#
#     rho1, theta1 = line1[0]
#     rho2, theta2 = line2[0]
#
#     a = np.array([
#         [np.cos(theta1), np.sin(theta1)],
#         [np.cos(theta2), np.sin(theta2)]
#     ])
#     b = np.array([[rho1], [rho2]])
#
#     intersection = [np.round(val) for val in np.linalg.solve(a, b)]
#
#     return intersection
#
#
# def find_all_intersections(image, vertical_lines, horizontal_lines):
#     # Get image width & height
#     height, width = image.shape[:2]
#
#     intersections = []
#
#     for vertical_line in vertical_lines:
#         for horizontal_line in horizontal_lines:
#             x1, y1 = find_intersection(vertical_line, horizontal_line)
#             # Validate intersection
#             if 0 <= x1 <= width and 0 <= y1 <= height:
#                 cv2.circle(image, (x1[0], y1[0]), 10, (255,0,0), 10)
#                 intersections.append((x1, y1))
#
#     return intersections
#
#
# def plot_lines(image, lines):
#     # Formula: ρ = x * cosθ + y * sinθ
#     # Need 2 points of x & y to form a line
#     # x = cosθ * ρ + y * sinθ
#     # y = sinθ * ρ + x * cosθ
#     # Subject to changes since y axis is inverted
#
#     # Get image width & height
#     height, width = image.shape[:2]
#
#     # Finding reference point
#     image_point = height if height > width else width
#
#     for line in lines:
#         rho, theta = line[0]
#         x1 = int(np.cos(theta) * rho + (-image_point) * np.sin(theta))
#         y1 = int(np.sin(theta) * rho + (-image_point) * -np.cos(theta))
#         x2 = int(np.cos(theta) * rho + image_point * np.sin(theta))
#         y2 = int(np.sin(theta) * rho + image_point * -np.cos(theta))
#
#         cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


def main():
    # Load image
    image = cv2.imread(filename="images/test.png", flags=cv2.IMREAD_COLOR)

    # Convert image to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding, invert resulting image for morphological operations
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 7)

    # Contour detection
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find the contours representing the numbers only
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) < 1000]

    # Draw contours over numbers on the grid
    cv2.drawContours(thresh, cnts, -1, (0, 0, 0), -1)

    # Define vertical kernel to fix lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    # Define horizontal kernel to fix lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

    # Closing to fix noise in grid lines
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=10)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=10)

    # Invert the image back to normal
    thresh = cv2.bitwise_not(thresh)

    # Contour detection again, this time we are extracting the grid
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Do a check if grid is fully extracted, no missing, no duplicates etc
    if len(cnts) == 81:
        grid_contour = [[] for x in range(0, 9)]

        # Sort contours (From top to bottom and left to right)
        cnts, _ = contours.sort_contours(cnts, method="top-to-bottom")

        # Extract every row
        for row_index, row in enumerate(range(0, 81, 9), start=0):
            row_cnts, _ = contours.sort_contours(cnts[row:row+9], method="left-to-right")
            for box_cnt in row_cnts:
                # Append
                grid_contour[row_index].append(box_cnt)

        for row in grid_contour:
            for box in row:
                cv2.drawContours(image, [box], -1, (0, 0, 0), 10)
                # Show image
                grid = cv2.resize(image, (600, 600))
                cv2.imshow("grid", grid)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
