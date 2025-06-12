import cv2
import os
from image_processing import get_grid_dimensions, filter_non_square_contours, transform_grid, reduce_noise

def main():
    """
    Loops through all unsolved puzzle images to extract all grid cells and save them in a directory defined
    """
    image_directory = r"C:\Users\Aozy\Desktop\sudoku_images"
    output_dir = r"cells"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    for file_name in os.listdir(image_directory):
        # Skip non-image files
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in valid_exts:
            continue

        input_path = os.path.join(image_directory, file_name)
        # Load image
        image = cv2.imread(filename=input_path, flags=cv2.IMREAD_COLOR)

        # Skip if image failed to load
        if image is None:
            print(f"Warning: could not read {input_path}, skipping.")
            continue

        # Extract grid
        grid_coordinates = get_grid_dimensions(image)

        # Check if grid is found
        if grid_coordinates is None:
            print(f"No grid detected in {file_name}, skipping.")
            continue

        # Crop grid with transformation
        grid = transform_grid(image, grid_coordinates)

        # Get grid dimensions
        grid_height, grid_width = grid.shape[:2]
        grid_area = grid_width * grid_height

        # Image preprocessing, reduce noise such as numbers/dots
        thresh = reduce_noise(grid)

        # Contour detection to extract the cells
        cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out non-square contours
        cnts = filter_non_square_contours(cnts)

        # Extract and save all valid cell ROIs
        for index, cnt in enumerate(cnts):
            x, y, width, height = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Threshold for cell size relative to grid
            if grid_area * (0.5/81) < area < grid_area * (1.5/81):
                roi = grid[y:y + height, x:x + width]
                out_name = f"{os.path.splitext(file_name)[0]}_{index}{ext}"
                out_path = os.path.join(output_dir, out_name)
                print(f"Saving {out_path}")
                cv2.imwrite(out_path, roi)

if __name__ == '__main__':
    main()
