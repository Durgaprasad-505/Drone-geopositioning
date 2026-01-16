#!/usr/bin/env python3
"""
Simple script to process training and test datasets.

Usage:
    python process_dataset.py                  # Process both train and test
    python process_dataset.py --train-only     # Process only training
    python process_dataset.py --test-only      # Process only test
"""

import os
import sys
import argparse
from drone_geopositioning import DroneGeopositioning, calculate_metrics, print_metrics_table


# Configure paths - UPDATE THESE TO YOUR PATHS
SATELLITE_MAP = 'data/map.tif'
TRAIN_DIR = 'data/drone_images/train'
TEST_DIR = 'data/drone_images/test'
GROUND_TRUTH = 'data/ground_truth.csv'
RESULTS_DIR = 'results'


def main():
    parser = argparse.ArgumentParser(description='Process drone images for geopositioning')
    parser.add_argument('--train-only', action='store_true', help='Process only training set')
    parser.add_argument('--test-only', action='store_true', help='Process only test set')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced version')
    parser.add_argument('--min-inliers', type=int, default=15, help='Minimum inliers')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(SATELLITE_MAP):
        print(f"ERROR: Satellite map not found at {SATELLITE_MAP}")
        print("Please update the SATELLITE_MAP path in this script.")
        sys.exit(1)
    
    # Initialize system
    print("\n" + "="*70)
    print("Drone Geopositioning - Batch Processing")
    print("="*70 + "\n")
    
    if args.enhanced:
        print("Using Enhanced version with improved matching...")
        from enhanced_geopositioning import EnhancedDroneGeopositioning
        geo_system = EnhancedDroneGeopositioning(
            satellite_map_path=SATELLITE_MAP,
            min_inliers=args.min_inliers,
            use_coarse_search=True
        )
    else:
        print("Using Standard version...")
        geo_system = DroneGeopositioning(
            satellite_map_path=SATELLITE_MAP,
            min_inliers=args.min_inliers
        )
    
    print()
    
    # Process training set
    if not args.test_only:
        if os.path.exists(TRAIN_DIR):
            print("\n" + "="*70)
            print("PROCESSING TRAINING SET")
            print("="*70)
            
            train_output = os.path.join(RESULTS_DIR, 'train_output.csv')
            
            train_results = geo_system.process_dataset(
                drone_images_dir=TRAIN_DIR,
                ground_truth_path=GROUND_TRUTH if os.path.exists(GROUND_TRUTH) else None,
                output_csv_path=train_output
            )
            
            # Calculate and display metrics
            print("\n" + "="*70)
            print("TRAINING SET METRICS")
            print("="*70)
            
            metrics = calculate_metrics(train_results)
            print_metrics_table(metrics)
            
            # Save metrics
            import pandas as pd
            metrics_df = pd.DataFrame([metrics])
            metrics_path = os.path.join(RESULTS_DIR, 'train_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            print(f"\nMetrics saved to: {metrics_path}")
            
            # Show sample results
            print("\nSample Results:")
            print(train_results[['image_name', 'latitude', 'longitude', 'success', 'inliers']].head(10))
            
        else:
            print(f"\nWarning: Training directory not found at {TRAIN_DIR}")
    
    # Process test set
    if not args.train_only:
        if os.path.exists(TEST_DIR):
            print("\n" + "="*70)
            print("PROCESSING TEST SET")
            print("="*70)
            
            test_output = os.path.join(RESULTS_DIR, 'test_output.csv')
            
            test_results = geo_system.process_dataset(
                drone_images_dir=TEST_DIR,
                ground_truth_path=None,  # No ground truth for test
                output_csv_path=test_output
            )
            
            print("\n" + "="*70)
            print("TEST SET COMPLETE")
            print("="*70)
            print(f"\nResults saved to: {test_output}")
            print(f"Total images processed: {len(test_results)}")
            print(f"Successful registrations: {test_results['success'].sum()}")
            
            # Show sample
            print("\nSample Results:")
            print(test_results[['image_name', 'latitude', 'longitude', 'success']].head(10))
            
        else:
            print(f"\nWarning: Test directory not found at {TEST_DIR}")
    
    # Final summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\nResults saved in: {RESULTS_DIR}/")
    print("\nFiles created:")
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith('.csv') or filename.endswith('.png'):
            filepath = os.path.join(RESULTS_DIR, filename)
            size = os.path.getsize(filepath)
            print(f"  - {filename} ({size:,} bytes)")
    
    print("\nNext steps:")
    print("  1. Review train_metrics.csv for performance evaluation")
    print("  2. Check error_analysis.png for error distribution")
    print("  3. Submit test_output.csv as your final predictions")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
