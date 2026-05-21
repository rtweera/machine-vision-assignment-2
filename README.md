# Machine Vision Assignment 2

A comprehensive Python project implementing solutions to machine vision problems focusing on line fitting algorithms, camera calibration and measurement, and image manipulation techniques.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Question 1: Line Fitting with TLS and RANSAC](#question-1-line-fitting-with-tls-and-ransac)
- [Question 2: Camera Calibration and Measurement](#question-2-camera-calibration-and-measurement)
- [Question 3: Perspective Warping and Image Compositing](#question-3-perspective-warping-and-image-compositing)
- [Technical Details](#technical-details)
- [Dependencies](#dependencies)

## Overview

This project solves three machine vision problems:

1. **Line Fitting**: Fit multiple lines to noisy data using Total Least Squares (TLS) and RANdom SAmple Consensus (RANSAC)
2. **Camera Measurement**: Use thin lens model to measure real-world object sizes from camera images
3. **Perspective Warping**: Composite an image onto a background with perspective transformation and realistic blending

All solutions are implemented in Python with NumPy, OpenCV, and Matplotlib for scientific computing and visualization.

## Project Structure

```
machine-vision-assignment-2/
├── src/                        # Main source code
│   ├── q1.py                  # Line fitting (TLS & RANSAC)
│   ├── q2.py                  # Camera measurement (lens model)
│   └── q3.py                  # Image warping & compositing
├── data/
│   └── lines.csv              # Sample line data for Q1
├── assets/
│   ├── earrings.jpg           # Image for Q2 measurement
│   ├── turf.jpg               # Background for Q3
│   └── flag.png               # Overlay image for Q3
├── output/                     # Generated results
│   ├── q1.png                 # Q1 visualization
│   ├── q2.png                 # Q2 visualization
│   ├── q3-*.png/jpg           # Q3 intermediate & final results
│   └── *.txt                  # Numerical outputs
├── docs/                       # Assignment documents
├── pyproject.toml             # Poetry project configuration
├── poetry.lock                # Dependency lock file
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)

### Setup Steps

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd machine-vision-assignment-2
   ```

2. **Install dependencies using Poetry**:
   ```bash
   poetry install
   ```
   
   This will create a virtual environment and install all required packages:
   - NumPy 2.4.2+
   - Pandas 3.0.1+
   - Matplotlib 3.10.8+
   - Seaborn 0.13.2+
   - OpenCV-Python 4.13.0.92+

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

## Quick Start

### Run All Questions
Execute each question script to generate visualizations and measurements:

```bash
# From the poetry shell
python src/q1.py  # Line fitting visualization
python src/q2.py  # Earring measurement (displays image window)
python src/q3.py  # Interactive perspective warping (requires mouse input)
```

### View Results
- Q1: Line fitting plots saved to `output/q1.png`
- Q2: Earring measurements and plots saved to `output/q2.png`
- Q3: Final composite image saved to `output/turf_with_flag.jpg`

## Question 1: Line Fitting with TLS and RANSAC

### Problem Statement
Given 300 noisy 2D points forming three roughly collinear clusters, fit:
- (a) A single line to the first 100 points using Total Least Squares (TLS)
- (b) Three lines to all 300 points using RANdom SAmple Consensus (RANSAC)

### Solution Approach

#### Part (a): Total Least Squares
TLS fits a line `ax + by + c = 0` by minimizing the orthogonal distance from points to the line.

**Algorithm**:
1. Center the data by subtracting the mean
2. Compute Singular Value Decomposition (SVD) of the centered points
3. The normal to the best-fit line is the last row of V^T (direction of least variance)
4. Normalize to unit length to get coefficients (a, b)
5. Compute c from the mean and normal vector

**Key Function**: `fit_line_tls(x, y)` returns `[a, b, c]` for line equation `ax + by + c = 0`

#### Part (b): RANSAC with Refinement
RANSAC robustly fits multiple lines by iteratively finding consensus among points.

**Algorithm**:
1. Randomly sample 2 points and fit a line through them
2. Count inliers (points within distance threshold)
3. If inlier count is best seen so far, save this line
4. Repeat for N iterations (default: 4000)
5. Refine using TLS on the best inliers
6. Remove inliers and repeat for next line
7. Continue until fewer than 35 inliers remain

**Parameters**:
- `n_iters`: Number of random samples (default: 4000)
- `threshold`: Distance threshold for inliers (default: 0.25 pixels)
- `min_inliers`: Minimum required inliers (default: 35)
- `seed`: Random seed for reproducibility

### Output

The script produces:
- Console output with line equations in both `ax + by + c = 0` and `y = mx + k` form
- Visualization with:
  - Left plot: TLS fit on single line data
  - Right plot: Three RANSAC lines with color-coded inliers and outliers marked in red

### Data Format
Input CSV file (`data/lines.csv`):
- 100 rows, 6 columns
- Columns 0-2: x-coordinates for lines 1-3
- Columns 3-5: y-coordinates for lines 1-3

### Running Q1
```bash
python src/q1.py
```

Output includes line equations and a matplotlib window displaying both fits.

---

## Question 2: Camera Calibration and Measurement

### Problem Statement
Given an image of earrings taken with a known camera setup, determine the physical dimensions of the earrings using the thin lens model and camera calibration parameters.

### Solution Approach

#### Thin Lens Model
Uses the thin lens equation to relate physical object size to sensor measurements:

**Key Formula**:
```
1/f = 1/D + 1/D'
```
where:
- `f` = focal length (8.0 mm)
- `D` = object distance from lens (720.0 mm)
- `D'` = image distance from lens (computed)

**Measurement Pipeline**:
1. Segment the earrings from background using HSV color space
2. Find the two largest connected components (the two earrings)
3. Extract bounding boxes in pixels
4. Convert pixels to sensor measurements (mm)
5. Use thin lens magnification to convert to object size

#### Calibration Parameters
```python
FOCAL_LENGTH_MM = 8.0          # Camera lens focal length
OBJECT_DISTANCE_MM = 720.0     # Distance from lens to subject
PIXEL_SIZE_UM = 2.2            # Sensor pixel size (2.2 × 2.2 μm)
```

#### Segmentation Strategy
Uses HSV color space to segment "gold-ish" colored earrings:
- Saturation threshold (> 20): Avoid pure white/grey regions
- Value threshold (< 250): Avoid bright background
- Morphological operations: Open to remove noise, close to fill gaps

#### Size Conversion
```
pixel_width → sensor_width (mm) → object_width (mm)
pixel_height → sensor_height (mm) → object_height (mm)

Average diameter = (avg_width + avg_height) / 2
```

### Output

The script produces:
- Console output with measurements:
  - Original bounding box dimensions (pixels)
  - Sensor-plane dimensions (mm)
  - Physical object dimensions (mm)
  - Average size across two earrings
- Visualization with three subplots:
  - Original image
  - Binary segmentation mask
  - Detected earrings with bounding boxes

### Running Q2
```bash
python src/q2.py
```

### Customization
To adjust earring detection for different images:
- Modify HSV thresholds in `segment_earrings()`
- Adjust morphological kernel size
- Change `min_area` threshold in `find_two_earrings()`

---

## Question 3: Perspective Warping and Image Compositing

### Problem Statement
Composite a flag image onto a turf background in perspective view, making it appear naturally placed and textured as if painted or printed on the surface.

### Solution Approach

#### Interactive Point Selection
1. User clicks 4 corner points on the image to define where the flag should be placed
2. Points are automatically ordered (top-left, top-right, bottom-right, bottom-left)
3. Perspective transformation matrix is computed using `cv2.getPerspectiveTransform()`

#### Image Transformation & Blending

**Perspective Warping**:
- Compute homography matrix H from flag coordinates to turf coordinates
- Warp flag using `cv2.warpPerspective()` with bilinear interpolation

**Feathering**:
- Create a mask from the warped flag region
- Apply Gaussian blur to mask edges for soft transitions
- Smoothly blend flag boundaries into the background

**Texture Imprinting**:
- Extract high-frequency texture from turf (grass texture)
- Modulate flag brightness by turf texture
- Creates realistic "painted on surface" effect

**Desaturation & Color Adjustment**:
- Reduce flag saturation (less neon appearance)
- Reduce flag brightness (more natural lighting)
- Controlled via `SAT_SCALE` and `VAL_SCALE` parameters

**Final Alpha Blending**:
```
output = turf * (1 - alpha*mask) + warped_flag * (alpha*mask)
```

#### Configuration Parameters
```python
ALPHA = 0.30           # Flag opacity (0=turf only, 1=flag only)
FEATHER = 31           # Edge softness (higher = softer)
ROTATE_FLAG_K = 1      # Rotation: 0=none, 1=90°CW, 2=180°, 3=270°CW

SAT_SCALE = 0.70       # Saturation reduction (0=grayscale, 1=original)
VAL_SCALE = 0.80       # Brightness reduction (0=black, 1=original)
TEXTURE_MIX = 1        # How much turf texture affects flag (0=none, 1=full)
GAUSS_BLUR = 5         # Blur amount for blending (0=none)
```

### Workflow

1. **Preparation**:
   - Load turf background and flag overlay images
   - Optionally rotate flag (90° increments)
   - Desaturate flag for more natural appearance

2. **Interactive Setup**:
   - Display turf image with mouse callback
   - User left-clicks to place 4 corner points
   - Press 'R' to reset, 'ESC' to quit after 4 points placed

3. **Transformation**:
   - Order points by angle from centroid (CCW ordering)
   - Compute perspective transformation matrix
   - Warp flag to destination quad

4. **Blending**:
   - Imprint turf texture onto flag
   - Apply feathering to mask edges
   - Alpha blend with turf background
   - Gaussian blur for natural appearance

5. **Output**:
   - Save composite image to `output/turf_with_flag.jpg`
   - Display result in window for inspection

### Running Q3
```bash
python src/q3.py
```

**Interaction**:
1. Window appears showing the turf image
2. Left-click 4 times to place corners where flag should go
3. Press ESC when done placing points
4. The script computes transformation and displays result
5. Composite image is saved to `output/turf_with_flag.jpg`

### Advanced Customization
- **Adjust opacity**: Change `ALPHA` (0.0 to 1.0)
- **Softer edges**: Increase `FEATHER` to odd numbers (15, 31, 51, etc.)
- **Different flag color**: Modify `SAT_SCALE` and `VAL_SCALE`
- **More realistic**: Increase `TEXTURE_MIX` or `GAUSS_BLUR`

---

## Technical Details

### Algorithms Used

#### Total Least Squares (TLS)
- Also known as Errors-in-Variables (EIV) model
- Minimizes orthogonal distance (perpendicular to line)
- Uses SVD for robust numerical computation
- More appropriate than OLS when both x and y have measurement error

#### RANSAC (RANdom SAmple Consensus)
- Robust fitting algorithm for data with outliers
- Iteratively samples minimal sets (2 points for line)
- Counts inliers (consensus) for each hypothesis
- Scales well: O(log(outlier_ratio))
- Post-refines using all inliers with TLS

#### Thin Lens Model
- Fundamental imaging equation: 1/f = 1/D + 1/D'
- Assumes ideal thin lens with negligible thickness
- Works for macro and micro photography
- Applicable to modern smartphone cameras and CCTV systems

#### Perspective Transformation
- 4-point correspondence determines unique 3×3 homography matrix
- Bilinear interpolation for smooth resampling
- Enables realistic 3D-like placement on planar surface

#### Image Blending
- Alpha compositing: standard transparency model
- Feathering: reduces visible edges via Gaussian blur
- Texture modulation: adds surface detail for realism

### Numerical Stability Considerations

1. **Line Normalization**: Lines stored as `[a, b, c]` with `a² + b² = 1`
2. **SVD**: More stable than eigendecomposition for TLS
3. **Homography**: Computed using OpenCV's robust algorithm
4. **Color Space**: HSV more intuitive for object segmentation than RGB
5. **Float32 Precision**: Sufficient for imaging operations; used for efficiency

---

## Dependencies

All dependencies are specified in `pyproject.toml`:

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | 2.4.2+ | Numerical computing, linear algebra |
| Pandas | 3.0.1+ | Data loading and manipulation |
| Matplotlib | 3.10.8+ | Visualization and plotting |
| Seaborn | 0.13.2+ | Statistical data visualization |
| OpenCV-Python | 4.13.0.92+ | Computer vision operations |
| Poetry | 2.0.0+ | Dependency management and packaging |

## Usage Notes

### Environment
- Requires Python 3.11 or higher (3.11, 3.12, or 3.13)
- All scripts expect to be run from repository root
- Asset paths are relative: `assets/`, `data/`, `output/`

### Output
- Matplotlib visualizations appear in separate windows
- Images are saved to `output/` directory
- Console output includes detailed measurements and equations

### Troubleshooting

**Issue**: Import errors
- Solution: Ensure you've run `poetry install` and activated the environment with `poetry shell`

**Issue**: File not found errors
- Solution: Run scripts from the repository root directory

**Issue**: Q2 segmentation not detecting earrings
- Solution: Adjust HSV thresholds in `segment_earrings()` function based on lighting

**Issue**: Q3 interactive window not responding
- Solution: Ensure OpenCV is properly installed; try updating with `poetry update`

## Author

Ravindu Weerasinghe  
Email: ravindutharuka2001@gmail.com

## License

This is an assignment submission. Please refer to course guidelines for usage rights.