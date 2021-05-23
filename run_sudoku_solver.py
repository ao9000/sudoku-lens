import cv2
import os
from image_processing import extract_grid, filter_non_square_contours, sort_grid_contours, reduce_noise
from digits_classifier.helper_functions import sudoku_cells_reduce_noise
import tensorflow as tf
from backtracking import backtracking
import numpy as np
import copy


def main():
    # Load trained model
    model = tf.keras.models.load_model('digits_classifier/models/model.h5')

    image_directory = "images"
    for file_name in os.listdir(image_directory):
        # Load image
        image = cv2.imread(filename=os.path.join(image_directory, file_name), flags=cv2.IMREAD_COLOR)

        # Extract grid
        grid = extract_grid(image)

        # Check if grid is found
        if grid is not None:
            # Image preprocessing, reduce noise such as numbers/dots, cover all numbers
            thresh = reduce_noise(grid)

            # Contour detection again, this time we are extracting the grid
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out non square contours
            cnts = filter_non_square_contours(cnts)

            # Do a check if grid is fully extracted, no missing, no duplicates etc
            if len(cnts) == 81:
                # Sort grid into nested list format
                grid_contours = sort_grid_contours(cnts)

                # Create a blank Sudoku board
                board = np.zeros((9, 9), dtype=int)
                # Run digit classifier
                for row_index, row in enumerate(grid_contours):
                    for box_index, box in enumerate(row):
                        # Extract ROI from contour
                        x, y, width, height = cv2.boundingRect(box)
                        roi = grid[y:y + height, x:x + width]

                        # Convert to greyscale & resize
                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        roi = cv2.resize(roi, (28,28), interpolation=cv2.INTER_NEAREST)

                        # Image thresholding & invert image
                        digit = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27,
                                                      11)

                        # Remove surrounding noise
                        digit = sudoku_cells_reduce_noise(digit)

                        # Digit present
                        if digit is not None:
                            # Reshape to fit model input
                            digit = digit.reshape((1, 28, 28, 1))

                            # Make prediction
                            board[row_index][box_index] = np.argmax(model.predict(digit), axis=-1)[0]+1

                # Perform backtracking
                solved_board, steps = backtracking(copy.deepcopy(board))

                print(f"{file_name}: Detected")

                # Check if valid puzzle
                if steps:
                    # Solved
                    print("Original")
                    print(board)

                    print(f"Solved in {steps} steps")
                    print(solved_board)

                else:
                    # Cannot be solved (Wrong/invalid puzzle)
                    print("Cannot be solved")

                # # Uncomment to save cells ROI
                # for row_index, row in enumerate(grid_contours):
                #     for box_index, box in enumerate(row):
                #         x, y, width, height = cv2.boundingRect(box)
                #         roi = grid[y:y + height, x:x + width]
                #         cv2.imwrite(f"digits_classifier/test/{file_name}[{row_index}][{box_index}].png", roi)

            else:
                print(f"{file_name}: Not detected")

        else:
            print("No Grid was found")


if __name__ == '__main__':
    main()
