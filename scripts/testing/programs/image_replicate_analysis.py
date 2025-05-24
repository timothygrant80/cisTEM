#!/usr/bin/env python3
"""
Image Replicate Analysis Tool

This script analyzes multiple replicate MRC images and calculates similarity metrics
between them. It accepts a file containing a list of image filenames and a threshold
value for calculating relative error metrics.

Example usage:
    python image_replicate_analysis.py --image-list /path/to/image_list.txt --threshold 6.90
"""

import sys
import os
import argparse

# Handle Python module import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

try:
    from cistem_test_utils.image_replicate_analysis import ImageReplicateAnalysis
except ImportError:
    # Fall back to annoying_hack.py approach for compatibility
    sys.path.insert(0, os.path.join(script_dir))
    import annoying_hack
    from cistem_test_utils.image_replicate_analysis import ImageReplicateAnalysis


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze replicate MRC images and calculate similarity metrics"
    )
    
    parser.add_argument(
        "--image-list", 
        required=True,
        help="Path to a file containing a list of MRC image filenames (one per line)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        required=True,
        help="Threshold value for relative error calculations (must be between 0 and 100)"
    )
    
    return parser.parse_args()


def read_image_list(file_path):
    """
    Read image filenames from a text file.
    
    Args:
        file_path: Path to the text file containing image filenames
        
    Returns:
        List of image filenames
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or contains invalid entries
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image list file not found: {file_path}")
        
    with open(file_path, 'r') as f:
        image_files = [line.strip() for line in f if line.strip()]
        
    if not image_files:
        raise ValueError(f"No image filenames found in {file_path}")
        
    # Check that all image files exist
    missing_files = [f for f in image_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"The following image files were not found: {', '.join(missing_files)}")
        
    return image_files


def main():
    """Main entry point for the script."""
    try:
        args = parse_args()
        
        # Read image filenames from the list file
        image_files = read_image_list(args.image_list)
        print(f"Found {len(image_files)} image files in {args.image_list}")
        
        # Create image replicate analyzer instance
        analyzer = ImageReplicateAnalysis(image_files, args.threshold)
        
        # Load images and verify dimensions
        print("Loading images and verifying dimensions...")
        analyzer.load_images()
        
        # Run the analysis
        print("Analyzing replicate images...")
        results = analyzer.analyze_replicates()
        
        # Print the results
        analyzer.print_analysis(results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())