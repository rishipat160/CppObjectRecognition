# C++ Object Recognition

A computer vision project that implements real-time object detection and classification using OpenCV and custom feature extraction. This project emphasizes manual implementation of core algorithms rather than relying on built-in OpenCV functions.

## Overview

This project demonstrates object recognition using traditional computer vision techniques. It processes video input to detect objects through thresholding, extracts shape-based features, and classifies objects using a nearest-neighbor approach with various distance metrics.

The implementation includes custom-built algorithms for:
- Adaptive thresholding with saturation adjustment (implemented from scratch)
- Morphological operations for noise reduction
- Connected component analysis for object detection
- Feature extraction (fill ratio, aspect ratio, Hu moments)
- Object classification with multiple distance metrics
- Real-time visualization of detected objects

Many core algorithms are implemented manually using direct matrix (Mat) manipulation rather than relying on OpenCV's built-in functions, providing deeper understanding of the underlying computer vision principles.

## Project Structure

- `src/`
  - `vidDisplay.cpp`: Main application for video capture and UI
  - `threshold.cpp`: Implementation of thresholding and feature extraction algorithms
- `include/`
  - `threshold.hpp`: Header file with function declarations and data structures
- `data/`
  - `object_features.csv`: Database of object features for classification
- `Makefile`: Build configuration

## Features

### Object Detection
- Custom grayscale conversion with saturation-based adjustment
- Manual thresholding implementation using pixel-by-pixel operations
- Custom morphological operations (erosion, dilation, opening, closing)
- Connected component analysis with size filtering

### Feature Extraction
- Percent filled (area / oriented bounding box area)
- Aspect ratio (ratio of the oriented bounding box dimensions)
- Hu moments (first two moments - rotation invariant shape descriptors)
- Center of mass (centroid)
- Principal axis orientation
- Oriented bounding box

### Classification
- Nearest neighbor classification
- Multiple distance metrics:
  - Euclidean distance
  - Scaled Euclidean distance
  - Cosine similarity
  - Scaled L1 (Manhattan) distance
- Confidence estimation

## Setup and Installation

### Prerequisites
- Windows with Visual Studio
- OpenCV 4.1.1 or later

### Building the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/CppObjectRecognition.git
   ```

2. Navigate to the project directory:
   ```bash
   cd CppObjectRecognition
   ```

3. Build the project using the provided Makefile:
   ```bash
   nmake -f Makefile
   ```

4. Run the application:
   ```bash
   nmake -f Makefile runVid
   ```

## Usage

### Controls
- `q`: Quit the application
- `s`: Save current frame and threshold image
- `n`: Add current object to database (prompts for label)
- `d`: Cycle through distance metrics
- `e`: Toggle evaluation mode for confusion matrix
- `1-5`: Select true object label in evaluation mode
- `t`: Test current object and update confusion matrix
- `p`: Print confusion matrix

### Adding New Objects
1. Place an object in the camera view
2. Press `n` to capture its features
3. Enter a label for the object when prompted

### Classification
- Objects are automatically classified when detected
- The classification, confidence percentage, and distance metric are displayed
- Different distance metrics can provide better results for different objects

## Results

The system can effectively recognize objects based on their shape features. Performance varies based on:
- Lighting conditions
- Object orientation
- Background complexity
- Distance metric used

The confusion matrix feature allows for quantitative evaluation of classification performance.

## Limitations and Future Work

- Current implementation relies on shape features only
- Performance depends on good thresholding
- Objects trained under bright lighting may be misclassified under dim lighting due to changes in thresholded region shapes
- Future improvements could include:
  - Better handling of lighting variations through adaptive preprocessing
  - Color-based features
  - Texture analysis
  - Machine learning approaches
  - Multi-view object recognition

## Author

Rishi Patel

*This project was completed as part of the course CS5100 at Northeastern University.*
