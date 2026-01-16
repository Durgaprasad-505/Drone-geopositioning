#!/usr/bin/env python3
"""
Diagnostic script to check GeoTIFF georeferencing and coordinate transformation.
"""

import rasterio
from rasterio.transform import rowcol, xy
import sys


def diagnose_geotiff(geotiff_path):
    """
    Diagnose GeoTIFF georeferencing and coordinate system.
    """
    print("="*70)
    print("GeoTIFF Diagnostic Tool")
    print("="*70)
    print(f"\nFile: {geotiff_path}\n")
    
    try:
        with rasterio.open(geotiff_path) as dataset:
            print("✓ GeoTIFF opened successfully\n")
            
            # Basic info
            print("Image Information:")
            print(f"  Size: {dataset.width} x {dataset.height} pixels")
            print(f"  Bands: {dataset.count}")
            print(f"  Data type: {dataset.dtypes[0]}")
            print()
            
            # Coordinate Reference System
            print("Coordinate Reference System (CRS):")
            print(f"  CRS: {dataset.crs}")
            print(f"  EPSG Code: {dataset.crs.to_epsg() if dataset.crs else 'None'}")
            print()
            
            # Geotransform
            print("Geotransform:")
            transform = dataset.transform
            print(f"  {transform}")
            print()
            
            # Bounds
            print("Bounds:")
            bounds = dataset.bounds
            print(f"  Left: {bounds.left}")
            print(f"  Bottom: {bounds.bottom}")
            print(f"  Right: {bounds.right}")
            print(f"  Top: {bounds.top}")
            print()
            
            # Test coordinate conversion
            print("Test Coordinate Conversions:")
            print("-" * 70)
            
            # Test center pixel
            center_x = dataset.width // 2
            center_y = dataset.height // 2
            
            print(f"\nCenter pixel: ({center_x}, {center_y})")
            
            # Convert using rasterio's xy function
            lon, lat = xy(transform, center_y, center_x)
            print(f"  Using rasterio xy(): lon={lon}, lat={lat}")
            
            # Check if this looks like lat/lon
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                print(f"  ✓ Coordinates look like lat/lon (WGS84)")
            else:
                print(f"  ⚠ WARNING: Coordinates don't look like lat/lon!")
                print(f"    They might be in a projected coordinate system")
                print(f"    CRS: {dataset.crs}")
                
                # Try to convert to lat/lon
                if dataset.crs and dataset.crs.to_epsg() != 4326:
                    from rasterio.warp import transform as warp_transform
                    lon_ll, lat_ll = warp_transform(
                        dataset.crs, 
                        'EPSG:4326', 
                        [lon], 
                        [lat]
                    )
                    print(f"\n  Converting to WGS84 (EPSG:4326):")
                    print(f"    Longitude: {lon_ll[0]}")
                    print(f"    Latitude: {lat_ll[0]}")
                    
                    if -90 <= lat_ll[0] <= 90 and -180 <= lon_ll[0] <= 180:
                        print(f"    ✓ Converted coordinates look correct!")
                        print(f"\n  SOLUTION: You need to transform coordinates to EPSG:4326")
                    else:
                        print(f"    ✗ Still doesn't look right")
            
            # Corner tests
            print(f"\nCorner pixel conversions:")
            corners = [
                ("Top-left", 0, 0),
                ("Top-right", dataset.width-1, 0),
                ("Bottom-left", 0, dataset.height-1),
                ("Bottom-right", dataset.width-1, dataset.height-1)
            ]
            
            for name, px, py in corners:
                lon, lat = xy(transform, py, px)
                print(f"  {name:15} ({px:5}, {py:5}): lon={lon:.6f}, lat={lat:.6f}")
            
            print()
            print("="*70)
            
            # Check against expected values
            print("\nExpected Range Check:")
            print("  Based on your ground truth, GPS should be around:")
            print("    Latitude: 13.025 to 13.027")
            print("    Longitude: 77.562 to 77.564")
            print()
            
            # Check if any corner is in range
            in_range = False
            for name, px, py in corners:
                lon, lat = xy(transform, py, px)
                if 13.02 <= lat <= 13.03 and 77.56 <= lon <= 77.57:
                    print(f"  ✓ {name} corner is in expected range!")
                    in_range = True
            
            if not in_range:
                print(f"  ⚠ WARNING: No corners are in the expected GPS range!")
                print(f"    This satellite map might:")
                print(f"      1. Be in a different projection (needs conversion)")
                print(f"      2. Cover a different geographic area")
                print(f"      3. Have incorrect georeferencing metadata")
            
            return dataset.crs
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        geotiff_path = sys.argv[1]
    else:
        print("Usage: python diagnose_geotiff.py <path_to_geotiff>")
        print("\nExample:")
        print("  python diagnose_geotiff.py data/map.tif")
        sys.exit(1)
    
    diagnose_geotiff(geotiff_path)
