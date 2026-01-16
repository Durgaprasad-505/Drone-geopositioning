#!/usr/bin/env python3
"""
Diagnostic script to check ground truth CSV format and compatibility.
"""

import pandas as pd
import sys
import os


def check_ground_truth(csv_path):
    """
    Check ground truth CSV file format and provide diagnostics.
    """
    print("="*70)
    print("Ground Truth CSV Diagnostic Tool")
    print("="*70)
    print(f"\nFile: {csv_path}\n")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found at {csv_path}")
        return False
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"✓ CSV loaded successfully")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print()
        
        # Show columns
        print("Available columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. '{col}' - Type: {df[col].dtype}")
        print()
        
        # Show sample data
        print("First 5 rows:")
        print(df.head())
        print()
        
        # Detect expected columns
        cols_lower = {col.lower(): col for col in df.columns}
        
        img_col = None
        lat_col = None
        lon_col = None
        
        # Find image name column
        for possible in ['image_name', 'imagename', 'image', 'filename', 'file_name', 'name', 'timestamp']:
            if possible in cols_lower:
                img_col = cols_lower[possible]
                break
        
        # Find latitude column
        for possible in ['latitude', 'lat', 'y']:
            if possible in cols_lower:
                lat_col = cols_lower[possible]
                break
        
        # Find longitude column
        for possible in ['longitude', 'lon', 'long', 'lng', 'x']:
            if possible in cols_lower:
                lon_col = cols_lower[possible]
                break
        
        print("Column Detection:")
        print(f"  Image column: {img_col if img_col else 'NOT FOUND ❌'}")
        print(f"  Latitude column: {lat_col if lat_col else 'NOT FOUND ❌'}")
        print(f"  Longitude column: {lon_col if lon_col else 'NOT FOUND ❌'}")
        print()
        
        if img_col and lat_col and lon_col:
            print("✓ All required columns detected!")
            print()
            
            # Special note for timestamp column
            if img_col and img_col.lower() == 'timestamp':
                print("📝 Note: Using 'timestamp' column for image matching")
                print("   Image filenames like '1399_663896512.png' will be matched to")
                print("   timestamp '1399.663896512' in the CSV (underscore → period)")
                print()
            
            # Check data types
            print("Data Type Validation:")
            
            # Check latitude
            try:
                lat_numeric = pd.to_numeric(df[lat_col], errors='coerce')
                nan_count = lat_numeric.isna().sum()
                if nan_count > 0:
                    print(f"  ⚠ Latitude: {nan_count} non-numeric values found")
                else:
                    print(f"  ✓ Latitude: All numeric (range: {lat_numeric.min():.6f} to {lat_numeric.max():.6f})")
            except Exception as e:
                print(f"  ❌ Latitude: Error checking - {e}")
            
            # Check longitude
            try:
                lon_numeric = pd.to_numeric(df[lon_col], errors='coerce')
                nan_count = lon_numeric.isna().sum()
                if nan_count > 0:
                    print(f"  ⚠ Longitude: {nan_count} non-numeric values found")
                else:
                    print(f"  ✓ Longitude: All numeric (range: {lon_numeric.min():.6f} to {lon_numeric.max():.6f})")
            except Exception as e:
                print(f"  ❌ Longitude: Error checking - {e}")
            
            # Check image names
            try:
                unique_images = df[img_col].nunique()
                total_images = len(df)
                print(f"  ✓ Images: {unique_images} unique out of {total_images} total")
                if unique_images < total_images:
                    print(f"    ⚠ Warning: {total_images - unique_images} duplicate image entries")
            except Exception as e:
                print(f"  ❌ Images: Error checking - {e}")
            
            print()
            print("✓ Ground truth CSV is compatible with the pipeline!")
            print()
            print("The pipeline will automatically detect these columns:")
            print(f"  - Image: '{img_col}'")
            print(f"  - Latitude: '{lat_col}'")
            print(f"  - Longitude: '{lon_col}'")
            
            return True
        else:
            print("❌ ERROR: Missing required columns")
            print()
            print("Your CSV must have columns for:")
            print("  1. Image names (e.g., 'image_name', 'filename', 'image')")
            print("  2. Latitude (e.g., 'latitude', 'lat', 'y')")
            print("  3. Longitude (e.g., 'longitude', 'lon', 'x')")
            print()
            print("Suggestions:")
            
            if not img_col:
                print(f"  - Add an image name column or rename existing column")
            if not lat_col:
                print(f"  - Add a latitude column or rename existing column")
            if not lon_col:
                print(f"  - Add a longitude column or rename existing column")
            
            return False
            
    except Exception as e:
        print(f"❌ ERROR loading CSV: {e}")
        return False
    
    finally:
        print("="*70)


def create_template():
    """
    Create a template ground truth CSV file.
    """
    template_data = {
        'image_name': ['image001.jpg', 'image002.jpg', 'image003.jpg'],
        'latitude': [37.7749, 37.7750, 37.7751],
        'longitude': [-122.4194, -122.4195, -122.4196]
    }
    
    template_df = pd.DataFrame(template_data)
    template_path = 'ground_truth_template.csv'
    template_df.to_csv(template_path, index=False)
    
    print(f"\n✓ Template created: {template_path}")
    print("\nTemplate format:")
    print(template_df.to_string(index=False))
    print("\nYou can use this as a reference for formatting your ground truth file.")


if __name__ == "__main__":
    print()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--template':
            create_template()
        else:
            csv_path = sys.argv[1]
            check_ground_truth(csv_path)
    else:
        print("Usage:")
        print("  python check_ground_truth.py <path_to_ground_truth.csv>")
        print("  python check_ground_truth.py --template")
        print()
        print("Examples:")
        print("  python check_ground_truth.py data/ground_truth.csv")
        print("  python check_ground_truth.py --template")
    
