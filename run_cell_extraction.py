"""
    Extract cell images from puzzles
    To ensure data quality, additional manual filtering of false positives may be needed
    Purpose of this script is to generate a training dataset for the digit classifier
"""


import cv2
import os
from image_processing import get_grid_dimensions, filter_non_square_contours, transform_grid, reduce_noise


def main():
    """
    Loops through all unsolved puzzle images to extract all grid cells and save them in a directory defined
    """
    image_directory = "images/unsolved"
    for file_name in os.listdir(image_directory):
        # Load image
        image = cv2.imread(filename=os.path.join(image_directory, file_name), flags=cv2.IMREAD_COLOR)

        # Extract grid
        grid_coordinates = get_grid_dimensions(image)

        # Check if grid is found
        if grid_coordinates is not None:
            # Crop grid with transformation
            grid = transform_grid(image, grid_coordinates)

            # Get grid dimensions
            grid_height, grid_width = grid.shape[:2]
            grid_area = grid_width * grid_height

            # Image preprocessing, reduce noise such as numbers/dots, cover all numbers
            thresh = reduce_noise(grid)

            # Contour detection again, this time we are extracting the grid
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out non square contours
            cnts = filter_non_square_contours(cnts)

            # Extract all cells detected, no matter correct or not
            for index, cnt in enumerate(cnts):
                x, y, width, height = cv2.boundingRect(cnt)
                roi = grid[y:y + height, x:x + width]
                area = cv2.contourArea(cnt)

                # Set threshold for cell to be valid
                if grid_area*(0.5/81) < area < grid_area*(1.5/81):
                    print(f"Saving {file_name}[{index}].png")
                    cv2.imwrite(f"cells/{file_name}[{index}].png", roi)


if __name__ == '__main__':
    main()
