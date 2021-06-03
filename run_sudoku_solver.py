"""
    Runs the whole pipeline of grid extraction, cell extraction, digit classification to backtracking

    Takes a unsolved sudoku puzzle image and outputs a solved sudoku puzzle image
"""

import cv2
import os
from image_processing import get_grid_dimensions, filter_non_square_contours, sort_grid_contours, reduce_noise, transform_grid
from digits_classifier.helper_functions import sudoku_cells_reduce_noise
import tensorflow as tf
from backtracking import backtracking, create_empty_board, BLANK_STATE
import numpy as np
import copy
import imutils


def main():
    """
    Loops through all unsolved sudoku puzzle images, and perform all operations from grid extraction, cell extraction,
    digit classification to backtracking to find the solution of the puzzle

    Once a solution is found, renders the answers on the unsolved sudoku image
    """
    # Load trained model
    model = tf.keras.models.load_model('digits_classifier/models/model.h5')

    image_directory = "images/unsolved"
    for file_name in os.listdir(image_directory):
        # Load image
        image = cv2.imread(filename=os.path.join(image_directory, file_name), flags=cv2.IMREAD_COLOR)

        # Check if image is too big
        # If so, Standardise image size to avoid error in cell image manipulation
        # Cells must fit in 28x28 for the model, big images will exceed this threshold with aspect ratio resize
        if image.shape[1] > 700:
            image = imutils.resize(image, width=700)

        # Extract grid
        grid_coordinates = get_grid_dimensions(image)

        # Check if grid is found
        if grid_coordinates is not None:
            # Crop grid with transformation
            grid = transform_grid(image, grid_coordinates)

            # Image preprocessing, reduce noise such as numbers/dots, cover all numbers
            thresh = reduce_noise(grid)

            # Contour detection again, this time we are extracting the grid
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out non square contours
            cnts = filter_non_square_contours(cnts)

            # Convert contours into data to work with
            # Do a check if grid is fully extracted, no missing, no duplicates etc
            if len(cnts) == 81:
                # Sort grid into nested list format same as sudoku
                grid_contours = sort_grid_contours(cnts)

                # Create a blank Sudoku board
                board = create_empty_board()

                # Run digit classifier
                for row_index, row in enumerate(grid_contours):
                    for box_index, box in enumerate(row):

                        # Extract cell ROI from contour
                        x, y, width, height = cv2.boundingRect(box)
                        roi = grid[y:y + height, x:x + width]

                        # Convert to greyscale
                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                        # Image thresholding & invert image
                        digit_inv = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 11)

                        # Remove surrounding noise
                        digit = sudoku_cells_reduce_noise(digit_inv)

                        # Digit present
                        if digit is not None:
                            # Reshape to fit model input
                            digit = digit.reshape((1, 28, 28, 1))

                            # Make prediction
                            board[row_index][box_index] = np.argmax(model.predict(digit), axis=-1)[0]+1

                # Perform backtracking to solve detected puzzle
                solved_board, steps = backtracking(copy.deepcopy(board))

                # Check if puzzle is valid
                if steps:
                    # Solved
                    # Draw answers on the sudoku image
                    for row_index, row in enumerate(board):
                        for box_index, box in enumerate(row):
                            # Filter for BLANK_STATES
                            if box == BLANK_STATE:
                                x, y, width, height = cv2.boundingRect(grid_contours[row_index][box_index])

                                # Calculate font size
                                for num in np.arange(1.0, 10.0, 0.1):
                                    text_size = cv2.getTextSize(str(solved_board[row_index][box_index]),
                                                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                                                fontScale=num, thickness=2)

                                    font_size = num
                                    if text_size[0][0] > width//2 or text_size[0][1] > height//2:
                                        break

                                # Fill in answers in sudoku image
                                cv2.putText(image, str(solved_board[row_index][box_index]),
                                            (x+grid_coordinates[0][0]+(width * 1//4),
                                             y+grid_coordinates[0][1]+(height * 3//4)),
                                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)

                    # Fill in information at bottom left
                    cv2.putText(image, f"Solved in {steps} steps",
                                (0, image.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)

                    # Save answers in solved directory
                    cv2.imwrite(f"images/solved/{os.path.splitext(file_name)[0]}.png", image)

                    print(f"File: {file_name}, Solved in {steps} steps")
                else:
                    # Cannot be solved (Wrong/invalid puzzle)
                    # Reasons can be invalid puzzle or grid/digits detected wrongly
                    print(f"File: {file_name}, Invalid puzzle or digit detection error")
            else:
                # Unable to tally 81 boxes
                print(f"File: {file_name}: Unable to detect 81 cells in grid")

        else:
            # Fail to detect grid
            print(f"File: {file_name}, Unable to detect grid")


if __name__ == '__main__':
    main()
