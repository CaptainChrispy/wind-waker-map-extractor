## Preface

This project marks my first experience with OpenCV for targeted image detection. The process proved both educational and rewarding, prompting me to document my approach for future reference and to help others exploring computer vision techniques.

I developed this script to extract assets from Wind Waker HD screenshots for the [Wind Waker Tracker app's](https://github.com/CaptainChrispy/wind-waker-tracker) interactive map feature. After encountering challenges with direct game asset extraction due to UI rendering methods, I opted to work with game screenshots instead. Rather than manually cropping of images, I utilized OpenCV to automate the process. This project demonstrates how to detect and extract the dotted minimap squares from Wind Waker HD's map interface for use in external applications.

## Table of Contents

- [Complete Processing Pipeline](#complete-processing-pipeline)
- [1. Image Loading and Preprocessing](#1-image-loading-and-preprocessing)
- [2. Edge Detection](#2-edge-detection)
- [3. Primary Method: Contour Detection](#3-primary-method-contour-detection)
- [4. Finding the Most Square-Like Polygon](#4-finding-the-most-square-like-polygon)
- [5. Filtering by Shape and Size](#5-filtering-by-shape-and-size)
- [6. Backup Method: Template Matching](#6-backup-method-template-matching)
- [7. Cropping and Saving the Image](#7-cropping-and-saving-the-image)

## Complete Processing Pipeline

![Complete Processing Pipeline](img/screenshot_all_steps.jpg)

## 1. Image Loading and Preprocessing
```python
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale for easier processing
```

![Original Image](img/screenshot_original.jpg)
![Grayscale Conversion](img/screenshot_grayscale.jpg)

When loading the image, I convert it to grayscale using OpenCV's [cvtColor transformation](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html). This simplifies the image by reducing it from a 3-channel color image to a single-channel intensity image. Grayscale conversion is beneficial because it:
- Reduces memory usage and computational complexity
- Simplifies edge detection by focusing on intensity changes rather than color changes
- Is often sufficient for shape detection tasks where color isn't important, such as this one

## 2. Edge Detection
```python
edges = cv2.Canny(gray_image, config.canny_threshold_low, config.canny_threshold_high)
```

![Edge Detection](img/screenshot_edges.jpg)

Next, I apply [Canny edge detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html), a multi-stage algorithm for detecting edges in images. It essentially works by taking the grayscale image and identifying areas with sharp transitions in pixel brightness, which correspond to edges.

The two threshold values control the sensitivity. Any edge above the higher threshold is considered a strong edge, while edges below the lower threshold are suppressed. Edges between these thresholds are kept only if they are connected to strong edges.

## 3. Contour Detection
```python
# Find contours from edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

![All Contours](img/screenshot_all_contours.jpg)

After edge detection, I use OpenCV's [contour finding function](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html) to identify the boundaries of shapes in the image. 

The key parameters in our implementation are:
- `cv2.RETR_EXTERNAL`: Retrieves only the outermost contours and ignores any nested contours.
- `cv2.CHAIN_APPROX_SIMPLE`: A compression method that stores only the endpoints of horizontal, vertical, and diagonal segments, significantly reducing memory usage without losing important shape information.

The Wind Waker HD's dotted border around the island forms a relatively clean contour after edge detection, making it ideal for this approach.

## 4. Finding the Most Square-Like Polygon
```python
best_square = None
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, config.epsilon * cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
```

![Potential Squares](img/screenshot_potential_squares.jpg)

To identify our target, I use the [Douglas-Peucker algorithm](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html) (implemented via `cv2.approxPolyDP`), which simplifies contours by reducing the number of points while preserving the essential shape. The epsilon parameter (`config.epsilon * cv2.arcLength(cnt, True)`) controls the approximation accuracy - smaller values result in more detailed approximations.

## 5. Filtering by Shape and Size
```python
# Check for square shape
if config.aspect_ratio_min < aspect_ratio < config.aspect_ratio_max and \
    config.width_min < w < config.width_max and \
    config.height_min < h < config.height_max:
    
    if w == config.target_width:
        return (x, y, w, h)
    elif best_square is None or abs(w - config.target_width) < abs(best_square[2] - config.target_width):
        best_square = (x, y, w, h)
```

![Selected Square](img/screenshot_selected_square.jpg)

This heuristic approach combines three criteria to identify our target square:

1. **Aspect ratio filtering**: Ensures the shape is approximately square, allowing for a 10% deviation from perfect 1:1 ratio to accommodate imperfect contour detection.

2. **Size constraints**: These values were determined by analyzing multiple Wind Waker HD screenshots. The consistency of the UI elements across the game makes this range reliable.

3. **Target width matching**: When multiple candidates meet the criteria, we prioritize the contour with a width closest to 319 pixels, which is the exact size observed in most 1920 x 1080 pixel screenshots. If we find an exact match with width = 319, we immediately select it and stop searching.

This multi-criteria approach creates a robust detection system that can handle variations in screenshots while maintaining precision.

## 6. Backup Method: Template Matching
```python
template = cv2.imread(config.template_path)
if template is None:
    print(f"{RED}Warning: Could not load template from {config.template_path}{RESET}")
    return None

gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

image_edges = cv2.Canny(gray_image, config.canny_threshold1, config.canny_threshold2)
template_edges = cv2.Canny(gray_template, config.canny_threshold1, config.canny_threshold2)

result = cv2.matchTemplate(image_edges, template_edges, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)

if max_val < config.match_threshold:
    return None

x, y = max_loc
h, w = template.shape[:2]
return (x, y, w, h)
```

When the primary contour detection method fails, the script falls back to [template matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html). This approach:

1. Uses a reference image (`ref.png`) of a clean dotted square
2. Applies edge detection to both the template and source image
3. Uses normalized cross-correlation to find the best match
4. Returns coordinates if the match quality exceeds a threshold

Template matching is useful for our case as the dotted square's outline and shade are consistent across screenshots, making it a reliable backup method, even if it means having a lower threshold for detection.

## 7. Cropping and Saving the Image
```python
x, y, w, h = best_square
padding = config.border_padding
cropped = image[y + padding:y + h - padding, x + padding:x + w - padding]
cv2.imwrite(output_path, cropped)
print(f'{GREEN}Cropped and saved to {output_path}{RESET}')
```

![Final Cropped Image](img/screenshot_cropped.jpg)

The final step uses Python's array slicing to crop the image and account for some padding to remove the dotted border around the map piece. The cropped image is then saved to the specified output path using OpenCV's `imwrite` function.

## Conclusion

This project demonstrates how combining relatively simple computer vision techniques can create a specialized tool for game screenshot processing. While this implementation is tailored for Wind Waker HD screenshots, the same principles can be applied to other applications by focusing on the specific characteristics of the target images.