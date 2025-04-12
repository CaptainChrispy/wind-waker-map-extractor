import cv2
import os
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration settings for cropping dotted square minimap images"""
    input_folder: str = "input_images"
    output_folder: str = "cropped_images"

    canny_threshold1: int = 50
    canny_threshold2: int = 150

    epsilon: float = 0.02

    aspect_ratio_min: float = 0.9
    aspect_ratio_max: float = 1.1
    width_min: int = 300
    width_max: int = 330
    height_min: int = 300
    height_max: int = 330
    target_width: int = 319

    border_padding: int = 5

    valid_extensions = ('.png', '.jpg', '.jpeg')


def crop_dotted_square(image_path, output_path, config=Config()):
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

    edges = cv2.Canny(gray, config.canny_threshold1, config.canny_threshold2) # Apply edge detection to find dotted border

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Look for the most square-like, medium-sized polygon
    best_square = None

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Check for square shape with reasonable size
            if config.aspect_ratio_min < aspect_ratio < config.aspect_ratio_max and config.width_min < w < config.width_max and config.height_min < h < config.height_max:
                if w == config.target_width:
                    best_square = (x, y, w, h)
                    break  # Found exact match, no need to continue
                # Otherwise keep the closest match to 319 pixels
                elif best_square is None or abs(w - config.target_width) < abs(best_square[2] - config.target_width):
                    best_square = (x, y, w, h)

    if best_square:
        x, y, w, h = best_square
        # Slight cropping to remove border
        padding = config.border_padding
        cropped = image[y + padding:y + h - padding, x + padding:x + w - padding]
        cv2.imwrite(output_path, cropped)
        print(f'Cropped and saved to {output_path}')
    else:
        print(f"No suitable square found in {image_path}")


def batch_crop_dotted_squares(input_folder, output_folder, config=Config()):
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
        if filename.lower().endswith(config.valid_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"cropped_{filename}")
            crop_dotted_square(input_path, output_path, config)


def main():
    config = Config()
    input_folder = config.input_folder
    output_folder = config.output_folder
    batch_crop_dotted_squares(input_folder, output_folder, config)

if __name__ == "__main__":
    main()