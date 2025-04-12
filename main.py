import cv2
import os
from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    """Configuration settings for cropping dotted square minimap images"""
    # File system settings
    input_folder: str = "input_images"
    output_folder: str = "cropped_images"
    valid_extensions: tuple = ('.png', '.jpg', '.jpeg')

    # Edge detection parameters
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    epsilon: float = 0.02

    # Square detection criteria
    aspect_ratio_min: float = 0.9
    aspect_ratio_max: float = 1.1
    width_min: int = 310
    width_max: int = 330
    height_min: int = 310
    height_max: int = 330
    target_width: int = 319

    # Template matching settings
    template_path: str = "ref.png"
    match_threshold: float = 0.3

    # Output image settings
    border_padding: int = 5

    # Debug settings
    debug: bool = False
    debug_folder: str = "debug_output"


GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def find_square_template(image: np.ndarray, config: Config) -> tuple:
    """
    Backup method using template matching to find the square.
    
    Args:
        image: Input image
        config: Configuration object
    
    Returns:
        tuple: (x, y, w, h) if square found, None otherwise
    """
    template = cv2.imread(config.template_path)
    if template is None:
        print(f"{RED}Warning: Could not load template from {config.template_path}{RESET}")
        return None
        
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < config.match_threshold:
        return None

    x, y = max_loc
    h, w = template.shape[:2]
    return (x, y, w, h)


def find_square_contours(image: np.ndarray, config: Config) -> tuple:
    """
    Tries to find the square using contour detection method.
    
    Args:
        image: Input image
        config: Configuration object
    
    Returns:
        tuple: (x, y, w, h) if square found, None otherwise
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, config.canny_threshold1, config.canny_threshold2)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_square = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if config.aspect_ratio_min < aspect_ratio < config.aspect_ratio_max and \
               config.width_min < w < config.width_max and \
               config.height_min < h < config.height_max:
                
                if w == config.target_width:
                    return (x, y, w, h)
                elif best_square is None or abs(w - config.target_width) < abs(best_square[2] - config.target_width):
                    best_square = (x, y, w, h)
        
    return best_square


def crop_dotted_square(image_path: str, output_path: str, config: Config = Config()) -> None:
    """
    Crops the dotted square minimap from the image and saves it to the output path.

    Args: 
        image_path: Path to the input image
        output_path: Path to save the cropped image
        config: Configuration object

    Returns:
        None
    """
    image = cv2.imread(image_path)
    best_square = find_square_contours(image, config)

    if not best_square:
        print(f"{YELLOW}No suitable square found in {image_path}{RESET}")
        print(f"{BLUE}Primary detection failed, trying template matching...{RESET}")

        best_square = find_square_template(image, config)
        if best_square:
            print(f"{GREEN}Template matching succeeded!{RESET}")
        else:
            print(f"{RED}Template matching failed. No square found.{RESET}")

            if config.debug:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, config.canny_threshold1, config.canny_threshold2)
                contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                save_debug_images(image_path, image, gray, edges, contours, config)

            return

    x, y, w, h = best_square
    padding = config.border_padding
    cropped = image[y + padding:y + h - padding, x + padding:x + w - padding]
    cv2.imwrite(output_path, cropped)
    print(f'{GREEN}Cropped and saved to {output_path}{RESET}')


def save_debug_images(image_path: str, image: np.ndarray, gray: np.ndarray, edges: np.ndarray, contours: list, config: Config) -> None:
    """
    Saves debug images when detection fails. Combines edge detection and contour visualization side by side.
    
    Args:
        image_path: Path to original image for filename extraction
        image: Original image
        gray: Grayscale image
        edges: Edge detection result
        contours: Detected contours
        config: Configuration object
    """
    filename = os.path.basename(image_path)
    debug_dir = config.debug_folder
    os.makedirs(debug_dir, exist_ok=True)

    debug_image = image.copy()
    cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
    
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    combined = np.hstack((edges_bgr, debug_image))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Edge Detection", (10, 30), font, 1, (0, 0, 0), 8)  # Black outline
    cv2.putText(combined, "Edge Detection", (10, 30), font, 1, (0, 255, 255), 2)  # Yellow text
    cv2.putText(combined, "Contour Detection", (image.shape[1] + 10, 30), font, 1, (0, 0, 0), 8)  # Black outline
    cv2.putText(combined, "Contour Detection", (image.shape[1] + 10, 30), font, 1, (0, 255, 255), 2)  # Yellow text
    
    cv2.imwrite(os.path.join(debug_dir, f"debug_{filename}"), combined)


def batch_crop_dotted_squares(input_folder: str, output_folder: str, config: Config = Config()) -> None:
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
    """Main entry point for the script."""
    config = Config(debug=True)
    batch_crop_dotted_squares(config.input_folder, config.output_folder, config)


if __name__ == "__main__":
    main()