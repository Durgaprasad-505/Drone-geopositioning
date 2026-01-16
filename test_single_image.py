#!/usr/bin/env python3
"""
Simple test script to verify GPS coordinate conversion is working correctly.
"""

import cv2
import sys
from drone_geopositioning import DroneGeopositioning
import rasterio
from pyproj import Transformer
import cv2
import numpy as np


def draw_location_on_map(
    map_path,
    lat,
    lon,
    output_path="output/map_with_point.png",
    radius=8
):
    """
    Draw estimated GPS location on the satellite map and save image.
    """

    # Open GeoTIFF
    with rasterio.open(map_path) as dataset:
        map_img = dataset.read([1, 2, 3])  # RGB
        transform = dataset.transform
        crs = dataset.crs

    # Convert to OpenCV format (HWC, BGR)
    map_img = np.transpose(map_img, (1, 2, 0))
    map_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR)

    # Convert WGS84 → Map CRS (EPSG:3857)
    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    x_m, y_m = transformer.transform(lon, lat)

    # Convert meters → pixel coordinates
    col, row = ~transform * (x_m, y_m)
    col, row = int(col), int(row)

    print(f"Drawing point at pixel: ({col}, {row})")

    # Draw point
    cv2.circle(
        map_img,
        (col, row),
        radius,
        (0, 0, 255),   # Red dot
        thickness=-1
    )

    # Optional label
    cv2.putText(
        map_img,
        f"({lat:.6f}, {lon:.6f})",
        (col + 10, row - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )

    # Save image
    cv2.imwrite(output_path, map_img)
    print(f"Saved map with point → {output_path}")


def test_single_image(map_path, image_path, expected_lat=None, expected_lon=None):
    """
    Test geopositioning on a single image and verify GPS coordinates.
    """
    print("="*70)
    print("Single Image Test")
    print("="*70)
    print(f"\nSatellite Map: {map_path}")
    print(f"Drone Image: {image_path}")
    if expected_lat and expected_lon:
        print(f"Expected GPS: ({expected_lat}, {expected_lon})")
    print()
    
    # Initialize system
    print("Initializing geopositioning system...")
    geo_system = DroneGeopositioning(map_path, min_inliers=15)
    print()
    
    # Load image
    print("Loading drone image...")
    drone_img = cv2.imread(image_path)
    if drone_img is None:
        print(f"ERROR: Could not load image from {image_path}")
        return False
    print(f"  Image size: {drone_img.shape[1]}x{drone_img.shape[0]}")
    print()
    
    # Estimate GPS
    print("Estimating GPS coordinates...")
    result = geo_system.estimate_gps(drone_img)
    print()
    
    # Display results
    print("="*70)
    print("RESULTS")
    print("="*70)
    
    if result['success']:
        est_lat = result['latitude']
        est_lon = result['longitude']
        
        print(f"✓ Registration successful!")
        print(f"\nEstimated GPS:")
        print(f"  Latitude:  {est_lat:.8f}")
        print(f"  Longitude: {est_lon:.8f}")
        print(f"\nMatch Quality:")
        print(f"  Inliers: {result['inliers']}")
        print(f"  Inlier Ratio: {result['inlier_ratio']:.1%}")
        print(f"  Execution Time: {result['execution_time']:.2f}s")
        
        # Sanity checks
        print(f"\nSanity Checks:")
        
        # Check if coordinates are in valid range
        if -90 <= est_lat <= 90:
            print(f"  ✓ Latitude in valid range [-90, 90]")
        else:
            print(f"  ✗ Latitude OUT OF RANGE: {est_lat}")
            print(f"    This suggests coordinate conversion issue!")
            
        if -180 <= est_lon <= 180:
            print(f"  ✓ Longitude in valid range [-180, 180]")
        else:
            print(f"  ✗ Longitude OUT OF RANGE: {est_lon}")
            print(f"    This suggests coordinate conversion issue!")
        
        # If expected coordinates provided, check proximity
        if expected_lat and expected_lon:
            error = geo_system.haversine_distance(
                est_lat, est_lon, expected_lat, expected_lon
            )
            print(f"\nError from Expected:")
            print(f"  Distance: {error:.2f} meters")
            
            if error < 20:
                print(f"  ✓ EXCELLENT: Error < 20m")
            elif error < 100:
                print(f"  ✓ GOOD: Error < 100m")
            elif error < 1000:
                print(f"  ⚠ ACCEPTABLE: Error < 1km")
            elif error < 10000:
                print(f"  ✗ POOR: Error < 10km - needs improvement")
            else:
                print(f"  ✗ FAILED: Error > 10km - likely matching wrong location")
        
        # Check if in Bangalore region (based on your ground truth)
        if 13.0 < est_lat < 13.05 and 77.5 < est_lon < 77.6:
            print(f"\n  ✓ Coordinates are in expected region (Bangalore area)")
        elif 12.5 < est_lat < 13.5 and 77.0 < est_lon < 78.0:
            print(f"\n  ⚠ Coordinates are near expected region but may be off")
        else:
            print(f"\n  ✗ Coordinates are FAR from expected region!")
            print(f"    Expected around: (13.026, 77.563)")
            print(f"    Got: ({est_lat}, {est_lon})")
        print("\nDrawing estimated location on map...")
        draw_location_on_map(
                map_path,
                est_lat,
                est_lon,
                output_path="output/map_with_estimated_location.png"
        )
        
        return True
        
    else:
        print(f"✗ Registration failed")
        print(f"  Reason: {result.get('error', 'Unknown')}")
        if 'inliers' in result:
            print(f"  Inliers found: {result['inliers']} (need {geo_system.min_inliers})")
        return False
    
    print("="*70)


if __name__ == "__main__":
    print()
    
    if len(sys.argv) < 3:
        print("Usage: python test_single_image.py <map.tif> <drone_image.png> [expected_lat] [expected_lon]")
        print("\nExample:")
        print("  python test_single_image.py data/map.tif data/train/image.png")
        print("  python test_single_image.py data/map.tif data/train/image.png 13.026787 77.563350")
        print()
        sys.exit(1)
    
    map_path = sys.argv[1]
    image_path = sys.argv[2]
    expected_lat = float(sys.argv[3]) if len(sys.argv) > 3 else None
    expected_lon = float(sys.argv[4]) if len(sys.argv) > 4 else None
    
    success = test_single_image(map_path, image_path, expected_lat, expected_lon)
    
    print()
    sys.exit(0 if success else 1)
