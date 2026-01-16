# 🚀 Drone Geopositioning - Complete Setup Guide

## 📦 Project Overview

This project implements drone GPS estimation by matching drone camera images with satellite maps using computer vision techniques (SIFT features, RANSAC, homography transformation).

---

## 📁 Project Structure

```
drone-geopositioning/
├── README.md                          # Complete documentation                   
├── requirements.txt                   # Python dependencies
│drone_geopositioning.py        # Main implementation
│enhanced_geopositioning.py     # Enhanced version (better matching)
│
├── Diagnostic Tools (Run Before Processing):
│   ├── diagnose_geotiff.py           # Check satellite map
│   ├── check_ground_truth.py         # Check ground truth CSV
│   └── test_single_image.py          # Test one image

```

---
### Step 1: Install Dependencies (2 minutes)

```bash
# Install Python packages
pip install -r requirements.txt

# Verify installation
python -c "import cv2, rasterio, pandas; print('✓ All packages installed')"
```

### Step 2: Organize Your Data (1 minute)

Put your data in this structure:

```
data/
├── map.tif                    # Your GeoTIFF satellite map
├── ground_truth.csv           # GPS coordinates for training images
└── drone_images/
    ├── train/                 # Training images (with ground truth)
    │   ├── 1399_663896512.png
    │   ├── 1400_123456789.png
    │   └── ...
    └── test/                  # Test images (no ground truth)
        ├── 1953_153540000.png
        ├── 1954_486108256.png
        └── ...
```

### Step 3: Run Diagnostics (5 minutes) ⚠️ CRITICAL

**This step is MANDATORY - it checks if your setup will work!**

```bash
# Check your satellite map
python diagnose_geotiff.py data/map.tif

# Check your ground truth CSV
python check_ground_truth.py data/ground_truth.csv

# Test one image
python test_single_image.py data/map.tif data/drone_images/train/1399_663896512.png 13.026787 77.563350
```

**Expected output from diagnostics:**
- ✓ GeoTIFF loads correctly
- ✓ Coordinates detected (timestamp, latitude, longitude)
- ✓ GPS values are in correct range (13.02-13.03, 77.56-77.57)

### Step 4: Choose Your Processing Method

Then follow the cells step by step.

**Option A: Python Script (Recommended for Production)**

```bash
python process_dataset.py
```

**Option B: Interactive Python**

```python
from drone_geopositioning import DroneGeopositioning

# Initialize
geo_system = DroneGeopositioning('data/map.tif')

# Process training set
train_results = geo_system.process_dataset(
    drone_images_dir='data/drone_images/train',
    ground_truth_path='data/ground_truth.csv',
    output_csv_path='results/train_output.csv'
)

# Process test set
test_results = geo_system.process_dataset(
    drone_images_dir='data/drone_images/test',
    ground_truth_path=None,
    output_csv_path='results/test_output.csv'
)
```

### Step 5: Check Results

Results will be in `results/` directory:
- `train_output.csv` - Training predictions with errors
- `test_output.csv` - Test predictions (submit this!)
- `train_metrics.csv` - Performance metrics
- `error_analysis.png` - Error distribution plot
- `success_analysis.png` - Success rate analysis
- `match_*.png` - Visualization of matches

---

## 📝 Detailed Instructions by Use Case

### For First-Time Users

1. Install dependencies
2. Run diagnostics (ALL THREE)
3. Open Jupyter notebook
4. Follow cells sequentially
5. Review results

### For Quick Testing

1. `python test_single_image.py data/map.tif data/drone_images/train/image.png`
2. Check if GPS coordinates look correct
3. If yes, proceed to full processing

### For Submitting Results

1. Process test images: `python process_dataset.py --test`
2. Check `results/test_output.csv` format:
   ```csv
   image_name,latitude,longitude
   test001.png,13.026787,77.563350
   ```
3. Submit this CSV file

---

## 🐛 Troubleshooting

### Issue: "KeyError: 'image_name'"
**Solution:** Your CSV uses different column names. Run:
```bash
python check_ground_truth.py data/ground_truth.csv
```

### Issue: GPS coordinates are huge numbers (1462789, 8634315)
**Solution:** Coordinate system issue. Run:
```bash
python diagnose_geotiff.py data/map.tif
```
The fixed code automatically handles this, but verify first.

### Issue: Median error > 1000 meters
**Solutions:**
1. Verify satellite map covers your flight area
2. Check coordinate conversion worked (see diagnose_geotiff.py output)
3. Try enhanced version for better matching
4. Read ACCURACY_FIX_GUIDE.md

### Issue: Success rate < 50%
**Solutions:**
1. Lower min_inliers: `DroneGeopositioning('map.tif', min_inliers=10)`
2. Increase features: Edit line 42 in drone_geopositioning.py
3. Use enhanced version with coarse-to-fine search


## 🔧 Configuration

### Quick Settings (in your script)

```python
# Initialize with custom settings
geo_system = DroneGeopositioning(
    satellite_map_path='data/map.tif',
    min_inliers=15  # Lower = more lenient (10-20)
)

# Adjust search parameters
geo_system.window_size = 512   # Larger = slower but may find larger features
geo_system.window_stride = 256  # Smaller = slower but more thorough
```

### Advanced Settings (edit drone_geopositioning.py)

Line 42-46: SIFT parameters
```python
self.feature_detector = cv2.SIFT_create(
    nfeatures=2000,          # More = slower but may help
    contrastThreshold=0.04,  # Lower = more features
    edgeThreshold=10
)
```

---

**For Results:**
- Training results: `results/train_output.csv`
- Test results: `results/test_output.csv` (submit this!)
- Metrics: `results/train_metrics.csv`

**Sample Results:**
             image_name   latitude  longitude  success
1    1953.153540000.png  13.026754  77.563375     True
2    1954.486108256.png  13.026740  77.563369     True
3    1955.824033440.png  13.026742  77.563374     True
4    1957.153702560.png  13.026730  77.563355     True
5    1958.486888000.png  13.026731  77.563357     True
6    1961.161126784.png  13.026724  77.563360     True
