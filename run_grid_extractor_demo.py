"""
    Shows how the grid cells recognition of a sudoku puzzle is done
"""


import cv2
import os
from image_processing import get_grid_dimensions, filter_non_square_contours, transform_grid, reduce_noise


def main():
    """
    Loops through all unsolved sudoku puzzle to perform grid cell extraction
    Showing which images are successful or unsuccessful and their extracted contours
    """
    # Initialize recording variables
    extracted = 0
    cell_error = 0
    grid_error = 0

    image_directory = "images/unsolved"
    
    for file_name in (dir_obj := os.listdir(image_directory)):
        # Load image
        image = cv2.imread(filename=os.path.join(image_directory, file_name), flags=cv2.IMREAD_COLOR)

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

            # Get len of cnts
            cnts_len = len(cnts)

            # Display original image
            # Process image
            image = cv2.resize(image, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
            image = cv2.putText(image, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Display
            cv2.imshow("Original image", image)
            cv2.waitKey(0)

            # Display end contour image
            # Process image
            contour_image = cv2.drawContours(grid, cnts, -1, (0, 255, 0), 2)
            contour_image = cv2.resize(contour_image, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
            contour_image = cv2.putText(contour_image, f"{cnts_len} detected contours", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Display the grid image
            cv2.imshow("Detected contours", contour_image)
            cv2.waitKey(0)

            # Close all images
            cv2.destroyAllWindows()

            if cnts_len == 81:
                print(f"File: {file_name}, Extracted successfully")
                extracted += 1
            else:
                print(f"File: {file_name}, Unable to extract grid cells properly")
                cell_error += 1
        else:
            print(f"File: {file_name}, Unable to detect grid")
            grid_error += 1

    # Print grid extraction results
    print(f"\nDetected successfully: {extracted}/{len(dir_obj)}")
    print(f"Unable to extract grid cells properly: {cell_error}")
    print(f"Unable to detect grid: {grid_error}")


if __name__ == '__main__':
    main()
