"""
Enhanced Drone Geopositioning with improved matching strategies.

This version includes:
1. Better preprocessing for cross-view matching
2. Multi-scale feature extraction
3. Coarse-to-fine search
4. GPS-guided search region (if approximate location known)
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from drone_geopositioning import DroneGeopositioning


class EnhancedDroneGeopositioning(DroneGeopositioning):
    """
    Enhanced version with better cross-view matching strategies.
    """
    
    def __init__(self, satellite_map_path: str, min_inliers: int = 15,
                 use_coarse_search: bool = True):
        """
        Initialize enhanced geopositioning system.
        
        Args:
            satellite_map_path: Path to GeoTIFF
            min_inliers: Minimum inliers for registration
            use_coarse_search: Enable coarse-to-fine search
        """
        super().__init__(satellite_map_path, min_inliers)
        self.use_coarse_search = use_coarse_search
        
        # Adjust parameters for better cross-view matching
        self.feature_detector = cv2.SIFT_create(
            nfeatures=3000,  # More features
            contrastThreshold=0.03,  # Lower threshold
            edgeThreshold=10,
            sigma=1.6
        )
        
        print("Enhanced geopositioning initialized")
        print(f"  Coarse-to-fine search: {use_coarse_search}")
    
    def preprocess_drone_image(self, drone_img: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for better cross-view matching.
        """
        # Convert to grayscale
        if len(drone_img.shape) == 3:
            gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = drone_img
        
        # Apply bilateral filter to reduce noise while keeping edges
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Sharpen the image slightly
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) / 9
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def coarse_search(self, drone_gray: np.ndarray, drone_kp: List, 
                      drone_desc: np.ndarray, stride_multiplier: int = 4) -> List:
        """
        Perform coarse search with larger stride to find candidate regions.
        
        Returns:
            List of (score, position) tuples for top candidates
        """
        h, w = self.satellite_gray.shape
        coarse_stride = self.window_stride * stride_multiplier
        
        candidates = []
        
        print(f"  Coarse search with stride {coarse_stride}...")
        
        for y in range(0, h - self.window_size + 1, coarse_stride):
            for x in range(0, w - self.window_size + 1, coarse_stride):
                # Extract window
                window = self.satellite_gray[y:y+self.window_size, x:x+self.window_size]
                
                # Extract features
                window_kp, window_desc = self.extract_features(window)
                
                if window_desc is None or len(window_kp) < 10:
                    continue
                
                # Match features
                matches = self.match_features(drone_desc, window_desc)
                
                if len(matches) >= 10:
                    # Score based on number of matches
                    score = len(matches)
                    candidates.append((score, (x, y), matches, window_kp))
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:10]  # Top 10 candidates
    
    def fine_search_around_candidate(self, drone_gray: np.ndarray, drone_kp: List,
                                     drone_desc: np.ndarray, candidate_pos: Tuple,
                                     search_radius: int = 512) -> Dict:
        """
        Perform fine search around a candidate position.
        """
        cx, cy = candidate_pos
        best_match = {
            'score': 0,
            'position': None,
            'homography': None,
            'inliers': 0,
            'inlier_ratio': 0.0,
            'matches': []
        }
        
        # Search in a region around the candidate
        x_start = max(0, cx - search_radius)
        y_start = max(0, cy - search_radius)
        x_end = min(self.satellite_gray.shape[1] - self.window_size, cx + search_radius)
        y_end = min(self.satellite_gray.shape[0] - self.window_size, cy + search_radius)
        
        for y in range(y_start, y_end + 1, self.window_stride):
            for x in range(x_start, x_end + 1, self.window_stride):
                # Extract window
                window = self.satellite_gray[y:y+self.window_size, x:x+self.window_size]
                
                # Extract features
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
                
                # Update best match
                if inliers > best_match['inliers'] and inliers >= self.min_inliers:
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
        
        return best_match
    
    def sliding_window_search(self, drone_gray: np.ndarray,
                             drone_kp: List, drone_desc: np.ndarray) -> Dict:
        """
        Enhanced search with coarse-to-fine strategy.
        """
        if self.use_coarse_search:
            # Step 1: Coarse search
            candidates = self.coarse_search(drone_gray, drone_kp, drone_desc,
                                          stride_multiplier=4)
            
            if not candidates:
                print("  No candidates found in coarse search, trying full search...")
                return super().sliding_window_search(drone_gray, drone_kp, drone_desc)
            
            print(f"  Found {len(candidates)} candidates, performing fine search...")
            
            # Step 2: Fine search around top candidates
            best_overall = {
                'score': 0,
                'position': None,
                'homography': None,
                'inliers': 0,
                'inlier_ratio': 0.0,
                'matches': []
            }
            
            for i, (score, pos, matches, window_kp) in enumerate(candidates[:5], 1):
                print(f"    Candidate {i}: {score} matches at {pos}")
                
                result = self.fine_search_around_candidate(
                    drone_gray, drone_kp, drone_desc, pos,
                    search_radius=512
                )
                
                if result['inliers'] > best_overall['inliers']:
                    best_overall = result
            
            print(f"  Best match: {best_overall['inliers']} inliers at {best_overall['position']}")
            return best_overall
        else:
            # Use standard sliding window
            return super().sliding_window_search(drone_gray, drone_kp, drone_desc)


def improve_matching_with_constraints(geo_system, ground_truth_df, search_radius_meters=100):
    """
    Helper function to constrain search using ground truth proximity.
    
    This can significantly improve results if you have approximate GPS from:
    - IMU/odometry
    - Previous GPS fix
    - Network-based location
    """
    # This would require modifications to constrain search region based on
    # expected GPS coordinates
    pass


if __name__ == "__main__":
    print("Enhanced Drone Geopositioning")
    print("=" * 70)
    print("\nThis enhanced version includes:")
    print("  - Better preprocessing (bilateral filter + sharpening)")
    print("  - More SIFT features (3000 vs 2000)")
    print("  - Coarse-to-fine search strategy")
    print("  - Fine search around promising candidates")
    print("\nUsage:")
    print("  from enhanced_geopositioning import EnhancedDroneGeopositioning")
    print("  geo_system = EnhancedDroneGeopositioning('map.tif')")
    print("  # ... rest same as before")
