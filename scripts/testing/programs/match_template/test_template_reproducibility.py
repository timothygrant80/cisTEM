#!/usr/bin/env python3
"""
Template Matching Reproducibility Test

This script runs the cisTEM template matching GPU binary multiple times using the same input data
and parameters, then analyzes the reproducibility by comparing the resulting MIP images.

It uses the Apoferritin dataset for testing, and saves the MIP (Maximum Intensity Projection)
files to a temporary directory with unique filenames for each replicate.

The script loads the MIP images using the mrcfile package and compares them using numpy to
measure pixel similarity between replicates with various metrics.

The script has two operational modes controlled by the FAST_DEVELOPMENT_CONDITIONS flag:
1. When True: Uses only binning=1.2 with 5.0 degree angular sampling (faster development mode)
2. When False: Tests all binning values (1.0, 1.2, 1.6) with 3.0 degree angular sampling

By default, this script outputs commands for the user to run the image_replicate_analysis.py
tool separately. Use the --run-analysis flag to automatically execute the analysis.
"""

import annoying_hack

from os.path import join, exists
from os import makedirs
import cistem_test_utils.args as tmArgs
import cistem_test_utils.make_tmp_runfile as mktmp
import cistem_test_utils.run_job as runner
from cistem_test_utils.temp_dir_manager import TempDirManager
from cistem_test_utils.threshold_utils import extract_threshold_value
from cistem_test_utils.image_replicate_analysis import ImageReplicateAnalysis
import os
import sys

# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'

# Define the number of replicates to run
NUM_REPLICATES = 5

# Define whether to use fast development conditions
FAST_DEVELOPMENT_CONDITIONS = False

# Define the binning values to test - modified based on development conditions
if FAST_DEVELOPMENT_CONDITIONS:
    BINNING_VALUES = [1.2]  # Only use 1.2 binning for faster development
else:
    BINNING_VALUES = [1.0, 1.2, 1.6]


def main():
    try:
        # Create a temp_dir_manager instance
        temp_manager = TempDirManager()
        
        # Parse command-line arguments
        args = tmArgs.parse_TM_args(wanted_binary_name)

        # Handle temp directory management options
        if args.list_temp_dirs:
            temp_manager.print_temp_dirs()
            return 0

        if args.rm_temp_dir is not None:
            success, message = temp_manager.remove_temp_dir(args.rm_temp_dir)
            print(message)
            return 0 if success else 1

        if args.rm_all_temp_dirs:
            success_count, failure_count = temp_manager.remove_all_temp_dirs()
            print(f"Successfully removed {success_count} temporary directories.")
            if failure_count > 0:
                print(f"Failed to remove {failure_count} temporary directories.")
            return 0 if failure_count == 0 else 1

        # Create a temporary directory to store our replicate MIPs and track it
        temp_dir = temp_manager.create_temp_dir(prefix="template_match_reproducibility_")
        print(f"Temporary directory created at: {temp_dir}")

        # Test each binning value
        for binning_value in BINNING_VALUES:
            print(f"\n{'-'*80}")
            print(f"Testing with binning value: {binning_value}")
            print(f"{'-'*80}")

            # Create a subdirectory for this binning value
            binning_dir = join(temp_dir, f"binning_{binning_value}")
            os.makedirs(binning_dir, exist_ok=True)

            # We'll run template matching for the defined number of replicates
            elapsed_time = [0] * NUM_REPLICATES
            mip_filenames = []
            hist_filenames = []
            threshold_values = []

            # Run the template matching for each replicate
            for replicate in range(NUM_REPLICATES):
                try:
                    # Use Apoferritin dataset with image 0
                    config = tmArgs.get_config(args, 'Apoferritin', 0, 0)
                    
                    # Set the angular sampling step based on development conditions
                    if FAST_DEVELOPMENT_CONDITIONS:
                        config['out_of_plane_angle'] = 5.0
                        config['in_plane_angle'] = 5.0
                    else:
                        config['out_of_plane_angle'] = 3.0
                        config['in_plane_angle'] = 3.0
                    
                    # Set the binning value
                    config['binning'] = binning_value

                    # Create a unique output file prefix for each replicate
                    original_prefix = config['output_file_prefix']
                    config['output_file_prefix'] = join(binning_dir, f"replicate_{replicate+1}")
                    makedirs(config['output_file_prefix'], exist_ok=True)

                    # Run the template matching
                    tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(config)
                    elapsed_time[replicate] = runner.run_job(tmp_filename_match_template)
                    runner.run_job(tmp_filename_make_template_results)

                    # The MIP file is already in our temp directory with the unique replicate prefix
                    mip_file = join(config['output_file_prefix'], 'mip.mrc')

                    # Check if MIP file exists
                    if not exists(mip_file):
                        raise FileNotFoundError(f"MIP file not found at {mip_file}")

                    # Get the histogram file path and extract the threshold value
                    hist_file = join(config['output_file_prefix'], 'hist.txt')
                    threshold_value = extract_threshold_value(hist_file)

                    mip_filenames.append(mip_file)
                    hist_filenames.append(hist_file)
                    threshold_values.append(threshold_value)

                    # Print threshold value only for the first replicate
                    if replicate == 0:
                        print(f"Threshold value: {threshold_value:.3f}")

                    print(f"Completed replicate {replicate+1}/{NUM_REPLICATES}, time: {elapsed_time[replicate]:.2f}s")

                except Exception as e:
                    print(f"Error during replicate {replicate+1}: {str(e)}")
                    continue

            # Check if we have at least 2 replicates
            if len(mip_filenames) < 2:
                print(f"Error: Only {len(mip_filenames)} replicates were successfully processed")
                print("At least 2 replicates are required for comparison. Exiting.")
                return 1

            # Verify all threshold values are the same
            if threshold_values:
                if not all(abs(v - threshold_values[0]) < 1e-10 for v in threshold_values):
                    print("Warning: Threshold values differ between replicates:")
                    for i, val in enumerate(threshold_values):
                        print(f"  Replicate {i+1}: {val:.6e}")

                # Use the first threshold value for calculations
                threshold_value = threshold_values[0]
            else:
                print("Error: No threshold values were extracted.")
                return 1

            print(f"Binning value: {binning_value}")

            # Create a list file for the MIP files
            mip_list_file = join(binning_dir, "mip_files.txt")

            if args.run_analysis:
                # Run the analysis automatically
                print(f"\n{'-'*80}")
                print(f"Running automatic analysis for binning {binning_value}")
                print(f"{'-'*80}")

                # Create the MIP files list
                import subprocess
                find_cmd = f"find {binning_dir} -name 'mip.mrc'"
                result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error creating MIP file list: {result.stderr}")
                    continue

                # Write the list to file
                with open(mip_list_file, 'w') as f:
                    f.write(result.stdout)

                # Verify we have files in the list
                if not result.stdout.strip():
                    print(f"No MIP files found in {binning_dir}")
                    continue

                print(f"Created MIP file list: {mip_list_file}")
                print(f"Found {len(result.stdout.strip().split())} MIP files")

                # Run the image replicate analysis
                try:
                    # Import and use the ImageReplicateAnalysis directly
                    image_files = result.stdout.strip().split('\n')
                    # Filter out empty lines
                    image_files = [f for f in image_files if f.strip()]

                    if len(image_files) < 2:
                        print(f"Error: Only {len(image_files)} MIP files found, need at least 2 for analysis")
                        continue

                    print(f"Running analysis on {len(image_files)} replicate MIP files...")
                    print(f"Using threshold: {threshold_value:.3f}")

                    # Create analyzer and run analysis
                    analyzer = ImageReplicateAnalysis(image_files, threshold_value)
                    analyzer.load_images()
                    results = analyzer.analyze_replicates()
                    analyzer.print_analysis(results)

                except Exception as e:
                    print(f"Error during image analysis: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                # Print commands for manual execution (original behavior)
                print(f"\n{'-'*80}")
                print(f"Replicate generation complete for binning {binning_value}")
                print(f"{'-'*80}")

                print("\nTo create a list of MIP files for analysis:")
                print(f"find {binning_dir} -name 'mip.mrc' > {mip_list_file}")

                print("\nOr to analyze other image types, like theta maps:")
                print(f"find {binning_dir} -name 'theta.mrc' > {binning_dir}/theta_files.txt")

                print("\nTo analyze the replicate results with the saved MIP list:")
                print(f"python3 scripts/testing/programs/image_replicate_analysis.py --image-list {mip_list_file} --threshold {threshold_value:.3f}")

                print("\nOr to extract the threshold value directly from histogram files:")
                print(f"threshold=$(awk '/threshold/{{print $NF; exit}}' {hist_filenames[0]})")
                print(f"python3 scripts/testing/programs/image_replicate_analysis.py --image-list {mip_list_file} --threshold $threshold")

        # Print the directory where files are saved
        print(f"\nMIP files saved in: {temp_dir}")
        if not args.run_analysis:
            print("To run automatic analysis next time, use: --run-analysis")
        print("To list temp directories: --list-temp-dirs")
        print("To remove this directory: --rm-temp-dir INDEX")
        print("To remove all temp directories: --rm-all-temp-dirs")

        return 0

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error in template matching reproducibility test: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


# Check if main function and run
if __name__ == '__main__':
    sys.exit(main())