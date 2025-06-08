import cv2
import numpy as np


def sudoku_cells_reduce_noise(digit_inv):
    """
    Noises in the cell images can impact the accuracy of the model
    Perform some image preprocessing on the extracted cell digits images to improve accuracy

    Process:
    1. Detect contour of the digit in the cell
    2. Filter the contours over5 pixels area
    3. Get the largest contour (Assuming the largest contour is the digit)
    4. Crop the digit
    5. Resize the digit to a standard size
    6. Create a black mat and paste the cropped digit onto it

    :param digit_inv: type: numpy.ndarray
    Binary image of the detected cell

    :return: type: numpy.ndarray
    Binary image of the digit after noise reduction
    """
    # Resize to 28x28 first
    height, width = digit_inv.shape
    # Use different interpolation methods based on enlargement or shrinking
    interp = (cv2.INTER_AREA if (height > 28 or width > 28) else cv2.INTER_CUBIC)
    # 28x28 resize
    small = cv2.resize(digit_inv, (28, 28), interpolation=interp)

    # Eliminate surrounding noise
    # Detect contours
    cnts, hierarchy = cv2.findContours(digit_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Relative area threshold
    # Total area of 28x28 image is 784 pixels
    total_area = 28 * 28
    # Calculate area threshold based on 5% of total area
    frac = 0.07
    area_thresh = total_area * frac
    # Filter contours over 5 pixel square area
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > area_thresh]

    # Check if any contour is detected
    if cnts:
        # Sort to largest contour (Digit)
        cnt = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]
        # Get coordinates, width, height of contour
        x, y, width, height = cv2.boundingRect(cnt)

        # Create buffer for crop
        crop_buffer = 1
        # Decrement crop buffer if buffer goes out of bounds of image
        while (y-crop_buffer) < 0 or (x-crop_buffer) < 0 or (y+height+crop_buffer) > digit_inv.shape[0] or (x+width+crop_buffer) > digit_inv.shape[1]:
            crop_buffer -= 1

            if crop_buffer == 0:
                break

        # Crop area
        digit_inv = digit_inv[y-crop_buffer:y + height+crop_buffer, x-crop_buffer:x + width+crop_buffer]
        # Update height & width
        height = height + (crop_buffer*2)
        width = width + (crop_buffer*2)

        # Create a black mat
        new_digit_inv = np.zeros((28, 28), np.uint8)

        # Standardize all image sizes
        # Maintain aspect ratio, resize via height or width (Whichever is bigger)
        resized_target_height_width = 17

        if height > width:
            # Height is larger
            aspect_ratio = resized_target_height_width / float(height)
            new_dimensions = (int(width * aspect_ratio), resized_target_height_width)
        else:
            # Width is larger
            aspect_ratio = resized_target_height_width / float(width)
            new_dimensions = (resized_target_height_width, int(height * aspect_ratio))

        # Don't allow any dimension to be 0, will result in error
        if new_dimensions[0] <= 3 or new_dimensions[1] <= 3:
            new_dimensions = (resized_target_height_width, resized_target_height_width)

        # Check if original image is larger is smaller
        if height > resized_target_height_width:
            # Shrink
            digit_inv = cv2.resize(digit_inv, new_dimensions, interpolation=cv2.INTER_AREA)
        else:
            # Expand
            digit_inv = cv2.resize(digit_inv, new_dimensions, interpolation=cv2.INTER_CUBIC)

        # Update width & height
        height, width = digit_inv.shape

        # Paste detected contour in the middle to center image
        new_digit_inv[14-height//2:14-height//2+height, 14-width//2:14-width//2+width] = digit_inv

        return new_digit_inv
    else:
        # No contour detected
        return None