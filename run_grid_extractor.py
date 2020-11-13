import cv2
import os
from image_processing import extract_grid, filter_non_square_contours, sort_grid_contours, reduce_noise


def main():
    image_directory = "images"
    for file_name in os.listdir(image_directory):
        # Load image
        print(file_name)
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
            print(len(cnts))
            if len(cnts) == 81:
                # Sort grid into nested list format
                grid_contours = sort_grid_contours(cnts)

                print(f"{file_name}: Detected")

                for row_index, row in enumerate(grid_contours):
                    for box_index, box in enumerate(row):
                        x, y, width, height = cv2.boundingRect(box)
                        roi = grid[y:y + height, x:x + width]
                        cv2.imwrite(f"digits_classifier/test/{file_name}[{row_index}][{box_index}].png", roi)

            else:
                print(f"{file_name}: Not detected")
                # for cnt in cnts:
                #     cv2.drawContours(grid, [cnt], -1, (0, 0, 0), 10)
                #     # Show image
                #     grid = cv2.resize(grid, (600, 600))
                #     cv2.imshow("grid", grid)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
        else:
            print("No Grid was found")


if __name__ == '__main__':
    main()
