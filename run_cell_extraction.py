import cv2
import os
from image_processing import extract_grid, filter_non_square_contours, sort_grid_contours, reduce_noise


def main():
    image_directory = "images"
    for file_name in os.listdir(image_directory):
        # Load image
        image = cv2.imread(filename=os.path.join(image_directory, file_name), flags=cv2.IMREAD_COLOR)

        # Extract grid
        grid = extract_grid(image)

        # Check if grid is found
        if grid is not None:
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
