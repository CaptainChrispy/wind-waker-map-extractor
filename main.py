import cv2
import os

def crop_dotted_square(image_path, output_path):
    """
    Crops the dotted square minimap from the image and saves it to the output path.

    Args: 
        image_path: Path to the input image
        output_path: Path to save the cropped image

    Returns:
        None
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale for easier processing

    edges = cv2.Canny(gray, 50, 150) # Apply edge detection to find dotted border

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Look for the most square-like, medium-sized polygon
    best_square = None
    best_area = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = w * h

            # Check for square shape and that its within expected area range
            if 0.9 < aspect_ratio < 1.1 and 300 < w < 330 and 300 < h < 330:
                if area > best_area:
                    best_area = area
                    best_square = (x, y, w, h)

    if best_square:
        x, y, w, h = best_square
        # Slight cropping to remove border
        padding = 5
        cropped = image[y + padding:y + h - padding, x + padding:x + w - padding]
        cv2.imwrite(output_path, cropped)
        print(f'Cropped and saved to {output_path}')
    else:
        print(f"No suitable square found in {image_path}")

def batch_crop_dotted_squares(input_folder, output_folder):
    """
    Batch processes images in the input folder to crop dotted squares and save them to the output folder.

    Args:
        input_folder: Folder containing input images
        output_folder: Folder to save cropped images

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"cropped_{filename}")
            crop_dotted_square(input_path, output_path)

input_folder = "input_images"
output_folder = "cropped_images"
batch_crop_dotted_squares(input_folder, output_folder)