"""
    Shows how the grid cells recognition of a sudoku puzzle is done
"""


import cv2
import os
import imutils
from image_processing import get_grid_dimensions, filter_non_square_contours, transform_grid, reduce_noise, get_cells_from_9_main_cells


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

            # Check how many valid cnts are found
            if 9 <= (cnts_len := len(cnts)) <= 90:
                # Salvageable
                if cnts_len == 81:
                    # All cells extracted, perfect
                    print(f"File: {file_name}, All 81 cells detected")
                    extracted += 1
                elif cnts_len == 9:
                    # Split main cells to 81 cells
                    cnts = get_cells_from_9_main_cells(cnts)

                    print(f"File: {file_name}, Main 9 cells detected")
                    extracted += 1
                else:
                    new_cnts = []

                    # In between, not sure if this is a valid grid
                    # Sort hierarchy, toss small contours to find main cells
                    # Only accept contours with hierarchy 0 (main contours)
                    # Format of hierarchy: [next, previous, child, parent]
                    for cnt, hie in zip(cnts, hierarchy[0]):
                        # Check if parent is -1 (Does not exist)
                        if hie[3] == -1:
                            new_cnts.append(cnt)

                    if len(new_cnts) == 9:
                        # Got all main cells
                        cnts = get_cells_from_9_main_cells(new_cnts)

                        print(f"File: {file_name}, Main 9 cells detected, with some noise")
                        extracted += 1
                    else:
                        # Unable to identify main cells
                        print(f"File: {file_name}, Unable to extract grid cells properly")
                        cell_error += 1

                # Update contour len, in case any contour filtering/adjustment was made
                cnts_len = len(cnts)
            else:
                # Unsalvageable
                print(f"File: {file_name}, Unable to extract grid cells properly")
                cell_error += 1

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
        else:
            print(f"File: {file_name}, Unable to detect grid")
            grid_error += 1

    # Print grid extraction results
    print(f"\nDetected successfully: {extracted}/{(dir_obj_len := len(dir_obj))}")
    print(f"Unable to extract grid cells properly: {cell_error}/{dir_obj_len}")
    print(f"Unable to detect grid: {grid_error}/{dir_obj_len}")


if __name__ == '__main__':
    main()
