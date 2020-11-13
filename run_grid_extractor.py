import cv2
from image_processing import extract_grid, filter_non_square_contours, sort_grid_contours, reduce_noise


def main():
    # Load image
    image = cv2.imread(filename="images/test2.jpg", flags=cv2.IMREAD_COLOR)

    # Extract grid
    grid = extract_grid(image)

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

        print("Detected")

        for row_index, row in enumerate(grid_contours):
            for box_index, box in enumerate(row):
                x, y, width, height = cv2.boundingRect(box)
                roi = grid[y:y + height, x:x + width]
                cv2.imwrite(f"digits_classifier/test/test7[{row_index}][{box_index}].png", roi)

    else:
        print("Not detected")
        for cnt in cnts:
            cv2.drawContours(grid, [cnt], -1, (0, 0, 0), 10)
            # Show image
            grid = cv2.resize(grid, (600, 600))
            cv2.imshow("grid", grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
