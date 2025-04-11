# Wind Waker Map Extractor

![Wind Waker Map Extraction Process](docs/img/banner.png)

A specialized tool that automatically extracts minimap squares from The Legend of Zelda: Wind Waker HD screenshots using computer vision.

## Overview

This tool leverages OpenCV to detect and extract the dotted square minimaps from Wind Waker HD's map screen. It was developed to assist with asset extraction for the [Wind Waker Tracker app](https://github.com/CaptainChrispy/wind-waker-tracker)'s interactive map feature.

## Key Features

- Automatic detection of dotted-border map squares using edge detection and contour analysis
- Batch processing for multiple screenshots 
- Clean cropping and edge removal with automatic padding

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- Screenshots must be 1920 x 1080 pixels in resolution

## Quick Start

1. **Installation**
   ```bash
   git clone https://github.com/CaptainChrispy/wind-waker-map-extractor.git
   cd wind-waker-map-extractor
   pip install opencv-python
   ```
2. **Processing Images**
   - Place your Wind Waker HD map screenshots in the `input_images` folder   git push --set-upstream origin master
   - Ensure all screenshots are 1920 x 1080 pixels in resolution (the tool is built for this size)
   - Run the script with `python main.py`
   - Find the processed images in the `cropped_images` folder

## How It Works

This tool uses computer vision techniques to identify and extract the dotted square minimaps from screenshots. The process automatically detects square borders and crops them to a consistent size.

For a detailed technical breakdown with visual examples of the 6-step pipeline, see the [technical documentation](docs/DOCUMENTATION.md).