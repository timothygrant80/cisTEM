"""
Image Replicate Analysis utilities for cisTEM tests.

This module provides a class for comparing multiple replicate MRC images and calculating
similarity metrics between them.
"""

import numpy as np
import mrcfile
import os
from typing import List, Dict, Tuple, Optional, Union


class ImageReplicateAnalysis:
    """Class for analyzing replicate MRC images and calculating similarity metrics."""

    def __init__(self, image_filenames: List[str], threshold_value: float = None):
        """
        Initialize the ImageReplicateAnalysis with a list of image filenames and threshold value.

        Args:
            image_filenames: List of MRC image files to analyze
            threshold_value: Threshold value for relative error calculations (must be between 0 and 100)

        Raises:
            ValueError: If threshold_value is not between 0 and 100
            ValueError: If fewer than 2 image filenames are provided
        """
        if len(image_filenames) < 2:
            raise ValueError("At least 2 image filenames are required for comparison")
        
        self.image_filenames = image_filenames
        
        # Validate threshold value if provided
        if threshold_value is not None:
            if not isinstance(threshold_value, (int, float)) or threshold_value <= 0 or threshold_value > 100:
                raise ValueError("Threshold value must be a positive number between 0 and 100")
        
        self.threshold_value = threshold_value
        self.image_data = []
        self.image_shapes = []
        self.image_dtypes = []
        
    def load_images(self) -> bool:
        """
        Load all image files and verify they have the same dimensions.
        
        Returns:
            bool: True if all images were loaded successfully with matching dimensions
            
        Raises:
            FileNotFoundError: If any image file cannot be found
            ValueError: If image dimensions do not match
        """
        self.image_data = []
        self.image_shapes = []
        self.image_dtypes = []
        
        # Load all images
        for filename in self.image_filenames:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Image file not found: {filename}")
                
            try:
                with mrcfile.open(filename) as mrc:
                    self.image_data.append(mrc.data)
                    self.image_shapes.append(mrc.data.shape)
                    self.image_dtypes.append(mrc.data.dtype)
            except Exception as e:
                raise IOError(f"Error loading {filename}: {str(e)}")
                
        # Check that all images have the same dimensions
        if len(set(str(shape) for shape in self.image_shapes)) > 1:
            raise ValueError(f"Image dimensions do not match: {self.image_shapes}")
            
        return True
        
    def analyze_replicates(self) -> Dict:
        """
        Analyze all replicate images and calculate similarity metrics.
        
        Returns:
            Dict: Dictionary containing pairwise and overall similarity metrics
        """
        if not self.image_data:
            self.load_images()
            
        num_replicates = len(self.image_data)
        
        # Generate all pairwise comparisons
        pairs = [(i, j) for i in range(num_replicates) for j in range(i+1, num_replicates)]
        
        results = {
            "num_replicates": num_replicates,
            "threshold_value": self.threshold_value,
            "pairwise_comparisons": [],
            "overall": {}
        }
        
        all_mean_abs_diffs = []
        
        # Calculate metrics for each pair of images
        for i, j in pairs:
            try:
                # Calculate mean absolute difference
                mean_abs_diff = np.mean(np.abs(self.image_data[i] - self.image_data[j]))
                all_mean_abs_diffs.append(mean_abs_diff)
                
                # Calculate relative error if threshold value is available
                if self.threshold_value and self.threshold_value > 0:
                    relative_error_ppm = (mean_abs_diff / self.threshold_value) * 1e6  # Parts per million
                else:
                    relative_error_ppm = None
                    
                comparison_result = {
                    "replicate_1": i + 1,
                    "replicate_2": j + 1,
                    "mean_abs_diff": mean_abs_diff,
                    "relative_error_ppm": relative_error_ppm
                }
                
                results["pairwise_comparisons"].append(comparison_result)
                
            except Exception as e:
                print(f"Error comparing replicates {i+1} and {j+1}: {str(e)}")
                
        # Calculate overall metrics across all comparisons
        if all_mean_abs_diffs:
            results["overall"]["mean_abs_diff_avg"] = np.mean(all_mean_abs_diffs)
            results["overall"]["mean_abs_diff_min"] = np.min(all_mean_abs_diffs)
            results["overall"]["mean_abs_diff_max"] = np.max(all_mean_abs_diffs)
            
            # Calculate average relative error if threshold is available
            if self.threshold_value and self.threshold_value > 0:
                results["overall"]["relative_error_ppm_avg"] = (np.mean(all_mean_abs_diffs) / self.threshold_value) * 1e6
                results["overall"]["relative_error_ppm_min"] = (np.min(all_mean_abs_diffs) / self.threshold_value) * 1e6
                results["overall"]["relative_error_ppm_max"] = (np.max(all_mean_abs_diffs) / self.threshold_value) * 1e6
                
        return results
        
    def print_analysis(self, results: Optional[Dict] = None) -> None:
        """
        Print the replicate analysis results in a formatted way.
        
        Args:
            results: Optional results dictionary from analyze_replicates().
                    If None, will run analyze_replicates() internally.
        """
        if results is None:
            results = self.analyze_replicates()
            
        num_replicates = results["num_replicates"]
        threshold_value = results["threshold_value"]
            
        print("\nReproducibility Analysis:")
        print("========================")
        print(f"Number of replicates analyzed: {num_replicates}")
        
        if threshold_value is not None:
            print(f"Threshold value: {threshold_value:.3f}")
            
        # Print pairwise comparisons
        for comparison in results["pairwise_comparisons"]:
            i = comparison["replicate_1"]
            j = comparison["replicate_2"]
            mean_abs_diff = comparison["mean_abs_diff"]
            relative_error_ppm = comparison["relative_error_ppm"]
            
            print(f"\nComparing replicate {i} vs {j}:")
            print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
            
            if relative_error_ppm is not None:
                print(f"  Relative error: {relative_error_ppm:.2f} ppm (relative to threshold value: {threshold_value:.3f})")
                
        # Print overall metrics
        if "overall" in results and results["overall"]:
            print("\nOverall reproducibility:")
            print(f"  Mean absolute diff (avg): {results['overall']['mean_abs_diff_avg']:.6f}")
            
            if threshold_value is not None:
                print(f"  Relative error (avg): {results['overall']['relative_error_ppm_avg']:.2f} ppm "
                      f"(relative to threshold value: {threshold_value:.3f})")