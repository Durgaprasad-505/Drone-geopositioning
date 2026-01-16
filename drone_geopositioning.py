
import os
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.warp import transform as warp_transform
import time
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DroneGeopositioning:
    
    def __init__(self, satellite_map_path: str, min_inliers: int = 15):
        self.satellite_map_path = satellite_map_path
        self.min_inliers = min_inliers
        self.satellite_img = None
        self.satellite_gray = None
        self.satellite_dataset = None
        self.transform = None
        
        # SIFT parameters - chosen for robustness to scale and rotation changes
        self.feature_detector = cv2.SIFT_create(
            nfeatures=3000,  # More features for difficult images
            contrastThreshold=0.02,  # Lower to get more features from low-texture images
            edgeThreshold=10  # Filter out edge-like features
        )
        
        # FLANN-based matcher for efficient matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Window search parameters
        self.window_size = 512  # Size of search windows on satellite map
        self.window_stride = 256  # Stride for sliding window
        
        print("Initializing Drone Geopositioning System...")
        self._load_satellite_map()
        
    def _load_satellite_map(self):
        """Load and preprocess the satellite map."""
        print(f"Loading satellite map from: {self.satellite_map_path}")
        
        # Load GeoTIFF with rasterio for georeferencing
        self.satellite_dataset = rasterio.open(self.satellite_map_path)
        self.transform = self.satellite_dataset.transform
        
        # Read image data
        satellite_data = self.satellite_dataset.read()
        
        # Handle different band configurations
        if satellite_data.shape[0] == 1:
            # Single band (grayscale)
            self.satellite_img = satellite_data[0]
        elif satellite_data.shape[0] >= 3:
            # Multi-band (RGB or more)
            # Convert from (bands, height, width) to (height, width, bands)
            self.satellite_img = np.transpose(satellite_data[:3], (1, 2, 0))
        
        # Convert to uint8 if needed
        if self.satellite_img.dtype != np.uint8:
            self.satellite_img = self._normalize_to_uint8(self.satellite_img)
        
        # Create grayscale version
        if len(self.satellite_img.shape) == 3:
            self.satellite_gray = cv2.cvtColor(self.satellite_img, cv2.COLOR_RGB2GRAY)
        else:
            self.satellite_gray = self.satellite_img
            
        print(f"Satellite map loaded: {self.satellite_gray.shape}")
        print(f"Map bounds: {self.satellite_dataset.bounds}")
        
    @staticmethod
    def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
        """Normalize image to uint8 range."""
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(img, dtype=np.uint8)
        return normalized
    
    def preprocess_drone_image(self, drone_img: np.ndarray) -> np.ndarray:
        # Convert to grayscale if needed
        if len(drone_img.shape) == 3:
            gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = drone_img
            
        # Apply CLAHE for better feature detection in varying lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def extract_features(self, img: np.ndarray) -> Tuple[List, np.ndarray]:

        keypoints, descriptors = self.feature_detector.detectAndCompute(img, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      ratio_threshold: float = 0.75) -> List:

        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        
        # Find k=2 nearest neighbors
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
                    
        return good_matches
    
    def find_homography_ransac(self, kp1: List, kp2: List, 
                               matches: List) -> Tuple[Optional[np.ndarray], int, float]:

        if len(matches) < 4:
            return None, 0, 0.0
        
        # Extract matched keypoint coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return None, 0, 0.0
        
        # Calculate inlier statistics
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(matches)
        
        return H, inliers, inlier_ratio
    
    def sliding_window_search(self, drone_gray: np.ndarray, 
                             drone_kp: List, drone_desc: np.ndarray) -> Dict:

        best_match = {
            'score': 0,
            'position': None,
            'homography': None,
            'inliers': 0,
            'inlier_ratio': 0.0,
            'matches': []
        }
        
        h, w = self.satellite_gray.shape
        drone_h, drone_w = drone_gray.shape
        
        # Calculate number of windows
        n_windows_y = (h - self.window_size) // self.window_stride + 1
        n_windows_x = (w - self.window_size) // self.window_stride + 1
        total_windows = n_windows_y * n_windows_x
        
        print(f"  Searching {total_windows} windows ({n_windows_y}x{n_windows_x})...")
        
        windows_checked = 0
        for y in range(0, h - self.window_size + 1, self.window_stride):
            for x in range(0, w - self.window_size + 1, self.window_stride):
                windows_checked += 1
                
                # Extract window
                window = self.satellite_gray[y:y+self.window_size, x:x+self.window_size]
                
                # Extract features from window
                window_kp, window_desc = self.extract_features(window)
                
                if window_desc is None or len(window_kp) < 10:
                    continue
                
                # Match features
                matches = self.match_features(drone_desc, window_desc)
                
                if len(matches) < 10:
                    continue
                
                # Find homography
                H, inliers, inlier_ratio = self.find_homography_ransac(
                    drone_kp, window_kp, matches
                )
                
                # Update best match based on inlier count
                if inliers > best_match['inliers'] and inliers >= self.min_inliers:
                    # Adjust keypoints to global coordinates
                    adjusted_kp = [cv2.KeyPoint(kp.pt[0] + x, kp.pt[1] + y, 
                                                kp.size, kp.angle, kp.response, 
                                                kp.octave, kp.class_id) 
                                 for kp in window_kp]
                    
                    best_match['score'] = inliers
                    best_match['position'] = (x, y)
                    best_match['homography'] = H
                    best_match['inliers'] = inliers
                    best_match['inlier_ratio'] = inlier_ratio
                    best_match['matches'] = matches
                    best_match['window_kp'] = adjusted_kp
                    
        print(f"  Best match: {best_match['inliers']} inliers at position {best_match['position']}")
        return best_match
    
    def pixel_to_gps(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:

        from rasterio.warp import transform as warp_transform
        
        # Use rasterio to convert pixel to map coordinates
        lon, lat = xy(self.transform, pixel_y, pixel_x)
        
        # Check if we need to transform to WGS84
        if self.satellite_dataset.crs:
            crs_epsg = self.satellite_dataset.crs.to_epsg()
            
            # If not already in WGS84 (EPSG:4326), transform it
            if crs_epsg != 4326:
                print(f"  Converting from {self.satellite_dataset.crs} to WGS84")
                lon_arr, lat_arr = warp_transform(
                    self.satellite_dataset.crs,
                    'EPSG:4326',
                    [lon],
                    [lat]
                )
                lon, lat = lon_arr[0], lat_arr[0]
        
        return lat, lon
    
    def estimate_gps(self, drone_img: np.ndarray) -> Dict:

        start_time = time.time()
        
        # Preprocess
        drone_gray = self.preprocess_drone_image(drone_img)
        
        # Extract features
        print("  Extracting drone features...")
        drone_kp, drone_desc = self.extract_features(drone_gray)
        
        if drone_desc is None or len(drone_kp) < 500:
            return {
                'success': False,
                'error': f'Insufficient features in drone image ({len(drone_kp) if drone_kp else 0} keypoints, need 500+)',
                'execution_time': time.time() - start_time,
                'inliers': 0,
                'inlier_ratio': 0.0
            }
        
        print(f"  Found {len(drone_kp)} keypoints in drone image")
        
        # Search for best match
        best_match = self.sliding_window_search(drone_gray, drone_kp, drone_desc)
        
        if best_match['homography'] is None or best_match['inliers'] < self.min_inliers:
            return {
                'success': False,
                'error': f'No reliable match found (inliers: {best_match["inliers"]})',
                'execution_time': time.time() - start_time,
                'inliers': best_match['inliers'],
                'inlier_ratio': best_match['inlier_ratio']
            }
        
        # Calculate drone center in satellite coordinates
        H = best_match['homography']
        drone_h, drone_w = drone_gray.shape
        drone_center = np.array([[[drone_w / 2, drone_h / 2]]], dtype=np.float32)
        
        # Transform to window coordinates
        sat_center_window = cv2.perspectiveTransform(drone_center, H)[0][0]
        
        # Adjust to global satellite coordinates
        window_x, window_y = best_match['position']
        sat_center_x = sat_center_window[0] + window_x
        sat_center_y = sat_center_window[1] + window_y
        
        # Convert to GPS
        lat, lon = self.pixel_to_gps(sat_center_x, sat_center_y)
        
        execution_time = time.time() - start_time
        
        return {
            'success': True,
            'latitude': lat,
            'longitude': lon,
            'pixel_x': sat_center_x,
            'pixel_y': sat_center_y,
            'inliers': best_match['inliers'],
            'inlier_ratio': best_match['inlier_ratio'],
            'execution_time': execution_time,
            'window_position': best_match['position'],
            'total_matches': len(best_match['matches'])
        }
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:

        R = 6371000  # Earth's radius in meters
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_phi / 2) ** 2 + 
             np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def process_dataset(self, drone_images_dir: str, 
                       ground_truth_path: Optional[str] = None,
                       output_csv_path: str = 'test_output.csv') -> pd.DataFrame:
        print(f"\n{'='*70}")
        print("Processing Drone Image Dataset")
        print(f"{'='*70}\n")
        
        # Load ground truth if provided
        ground_truth = None
        gt_img_col = None
        gt_lat_col = None
        gt_lon_col = None
        
        if ground_truth_path and os.path.exists(ground_truth_path):
            ground_truth = pd.read_csv(ground_truth_path)
            print(f"Loaded ground truth with {len(ground_truth)} entries")
            
            # Detect column names (case-insensitive)
            cols_lower = {col.lower(): col for col in ground_truth.columns}
            
            # Find image name column
            for possible in ['image_name', 'imagename', 'image', 'filename', 'file_name', 'name', 'timestamp']:
                if possible in cols_lower:
                    gt_img_col = cols_lower[possible]
                    break
            
            # Find latitude column
            for possible in ['latitude', 'lat', 'y']:
                if possible in cols_lower:
                    gt_lat_col = cols_lower[possible]
                    break
            
            # Find longitude column
            for possible in ['longitude', 'lon', 'long', 'lng', 'x']:
                if possible in cols_lower:
                    gt_lon_col = cols_lower[possible]
                    break
            
            if gt_img_col and gt_lat_col and gt_lon_col:
                print(f"Ground truth columns detected: {gt_img_col}, {gt_lat_col}, {gt_lon_col}")
            else:
                print(f"WARNING: Could not detect all ground truth columns")
                print(f"Available columns: {', '.join(ground_truth.columns)}")
                print(f"Detected - Image: {gt_img_col}, Lat: {gt_lat_col}, Lon: {gt_lon_col}")
                ground_truth = None
        
        # Get list of images
        image_files = sorted([f for f in os.listdir(drone_images_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(image_files)} drone images\n")
        
        results = []
        
        for idx, img_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {img_file}")
            print("-" * 70)
            
            # Load drone image
            img_path = os.path.join(drone_images_dir, img_file)
            drone_img = cv2.imread(img_path)
            
            if drone_img is None:
                print(f"  ERROR: Could not load image")
                results.append({
                    'image_name': img_file,
                    'latitude': None,
                    'longitude': None,
                    'success': False
                })
                continue
            
            # Estimate GPS
            result = self.estimate_gps(drone_img)
            
            # Prepare result entry
            result_entry = {
                'image_name': img_file,
                'latitude': result.get('latitude'),
                'longitude': result.get('longitude'),
                'success': result['success'],
                'inliers': result.get('inliers', 0),
                'inlier_ratio': result.get('inlier_ratio', 0.0),
                'execution_time': result.get('execution_time', 0.0)
            }
            
            # Calculate error if ground truth available
            if ground_truth is not None and result['success']:
                # Try direct match first
                gt_row = ground_truth[ground_truth[gt_img_col].astype(str) == img_file]
                
                # If timestamp column, try matching by extracting timestamp from filename
                if gt_row.empty and gt_img_col.lower() == 'timestamp':
                    # Extract timestamp from filename 
                    # Handle formats: "1399.663896512.png" or "1399_663896512.png"
                    img_timestamp = img_file.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                    img_timestamp = img_timestamp.replace('_', '.')  # Convert underscore to period
                    
                    # Try exact match
                    gt_row = ground_truth[ground_truth[gt_img_col].astype(str) == img_timestamp]
                    
                    # Try approximate match (floating point comparison)
                    if gt_row.empty:
                        try:
                            img_ts_float = float(img_timestamp)
                            # Find closest timestamp within 100ms (increased from 10ms)
                            time_diffs = abs(ground_truth[gt_img_col] - img_ts_float)
                            min_idx = time_diffs.idxmin()
                            if time_diffs[min_idx] < 0.1:  # Within 100ms
                                gt_row = ground_truth.loc[[min_idx]]
                                print(f"  Matched to timestamp: {ground_truth.loc[min_idx, gt_img_col]} (diff: {time_diffs[min_idx]*1000:.2f}ms)")
                        except (ValueError, TypeError):
                            pass
                
                if not gt_row.empty:
                    gt_lat = gt_row.iloc[0][gt_lat_col]
                    gt_lon = gt_row.iloc[0][gt_lon_col]
                    error = self.haversine_distance(
                        result['latitude'], result['longitude'],
                        gt_lat, gt_lon
                    )
                    result_entry['error_meters'] = error
                    print(f"  Localization Error: {error:.2f} meters")
                else:
                    print(f"  Warning: No ground truth found for {img_file}")
            
            results.append(result_entry)
            print(f"  Execution Time: {result.get('execution_time', 0):.2f} seconds")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save output CSV in required format
        output_df = results_df[['image_name', 'latitude', 'longitude']].copy()
        output_df.to_csv(output_csv_path, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_csv_path}")
        print(f"{'='*70}\n")
        
        return results_df


def calculate_metrics(results_df: pd.DataFrame) -> Dict:

    metrics = {}
    
    # Filter successful registrations
    successful = results_df[results_df['success'] == True]
    
    metrics['Total Images Processed'] = len(results_df)
    metrics['Registration Success Rate'] = (len(successful) / len(results_df) * 100) if len(results_df) > 0 else 0
    
    # Error-based metrics (only if ground truth available)
    if 'error_meters' in results_df.columns:
        errors = successful['error_meters'].dropna()
        
        if len(errors) > 0:
            metrics['Recall @ 5m (%)'] = (errors < 5).sum() / len(results_df) * 100
            metrics['Recall @ 20m (%)'] = (errors < 20).sum() / len(results_df) * 100
            metrics['Precision @ 5m (%)'] = (errors < 5).sum() / len(successful) * 100
            metrics['Precision @ 20m (%)'] = (errors < 20).sum() / len(successful) * 100
            metrics['Median Localisation Error'] = errors.median()
            metrics['Mean Localisation Error'] = errors.mean()
            metrics['Error Variance'] = errors.var()
            metrics['Error Std Dev'] = errors.std()
            metrics['Min Error'] = errors.min()
            metrics['Max Error'] = errors.max()
    
    # Inlier metrics
    if 'inlier_ratio' in results_df.columns and len(successful) > 0:
        metrics['Avg. Inlier Ratio'] = successful['inlier_ratio'].mean()
    
    # Execution time
    if 'execution_time' in results_df.columns and len(successful) > 0:
        metrics['Avg. Execution Time'] = successful['execution_time'].mean()
    
    return metrics


def print_metrics_table(metrics: Dict):
    """Print metrics in a formatted table."""
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'Time' in key:
                print(f"{key:.<50} {value:.3f} sec")
            elif '%' in key or 'Ratio' in key:
                print(f"{key:.<50} {value:.2f}%")
            else:
                print(f"{key:.<50} {value:.2f} m")
        else:
            print(f"{key:.<50} {value}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Drone Geopositioning System - Example Usage")
    print("=" * 70)
    print("\nThis script should be run with your dataset paths.")
    print("Example:")
    print("  python drone_geopositioning.py")
    print("\nPlease see the Jupyter notebook for full usage examples.")
