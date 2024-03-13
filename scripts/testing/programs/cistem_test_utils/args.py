import os
import sys
import argparse
import toml
# If we are in the container

default_data_dir = '/cisTEMdev/cistem_reference_images/TM_tests'


def get_config(args, data_dir: str, ref_number: int, img_number: int):

    if data_dir in ['Apoferritin', 'Kras']:
        if os.path.basename(args.test_data_path) != 'SPA':
            args.test_data_path = os.path.join(args.test_data_path, 'SPA')
    else:
        if data_dir not in ['Yeast', 'Crown', 'Lamella_from_je']:
            print('The data directory [' +
                  data_dir + '] does not seem to exist')
            print('Please provide a valid path to the test data directory as a second argument')
            sys.exit(1)

    config = toml.load(os.path.join(args.test_data_path,
                       data_dir, 'MetaData', data_dir + '.toml'))

    # FIXME: the image and ref number are annoying and should be more descriptive

    config['full_path_to_img'] = os.path.join(
        args.test_data_path, data_dir, 'Images', config.get('data')[img_number].get('img_name'))
    config['full_path_to_ref'] = os.path.join(
        args.test_data_path, data_dir, 'Templates', config.get('model')[ref_number].get('ref_name'))

    # confirm the pixel sizes match and set that in args
    if config.get('data')[img_number].get('pixel_size') != config.get('model')[ref_number].get('pixel_size'):
        print('The pixel sizes do not match between the image and reference')
        print('Please provide a valid path to the test data directory as a second arguments')
        sys.exit(1)
    else:
        config['pixel_size'] = config.get('data')[img_number].get('pixel_size')

    config['img_number'] = img_number
    config['ref_number'] = ref_number

    # Set some default search args that may be overwritten in a given test match_template
    config['out_of_plane_angle'] = 2.5
    config['in_plane_angle'] = 1.5
    config['defocus_range'] = 0
    config['defocus_step'] = 0
    config['pixel_size_range'] = 0
    config['pixel_size_step'] = 0
    config['padding_factor'] = 1.0
    config['mask_radius'] = 0
    config['max_threads'] = 2

    # some default search args that may be overwritten in a given test make_template_results
    config['results_mip_to_use'] = 'mip_scaled.mrc'
    config['result_min_peak_radius'] = 10.0
    config['result_number_to_process'] = 1
    config['sample_thickness'] = 2000.0  # Angstrom
    config['result_binning_factor'] = 4
    config['result_ignore_n_pixels_from_edge'] = -1

    for arg_val in args.args_to_check:
        config[arg_val] = getattr(args, arg_val)

    config['output_file_prefix'] = os.path.abspath(os.path.join(args.output_file_prefix, config.get('data')[img_number]['img_name']))
    os.makedirs(config['output_file_prefix'], exist_ok=True)

    return config


def parse_TM_args(wanted_binary_name):

    # Parse arguments requiring a path to the directory with the binary to be tested and optionally a path to the test data directory
    # We want to be able to modify these args so we'll make a copy of them in the config dict.

    # WARNING: any args added here must be copied to the config dict in get_config above
    # To help check, we'll keep this dict to use as a check.
    args_to_check = []
    parser = argparse.ArgumentParser(description='Test the k3 rotation binary')
    # Required argument
    parser.add_argument(
        '--binary_path', help='Path to the directory with the binary to be tested (Required)', required=True)
    args_to_check.append('binary_path')

    parser.add_argument('--test_data_path',
                        help='Path to the test data directory (Optional - defaults to /cisTEMdev/cistem_reference_images/TM_tests, then pwd)')
    args_to_check.append('test_data_path')

    # Argument for the output file path and prefix default to /tmp
    parser.add_argument('--output_file_prefix',
                        help='Path and prefix for the output files (Optional - defaults to /tmp)', default='/tmp')
    args_to_check.append('output_file_prefix')

    parser.add_argument('--gpu_idx',  default=0,
                        help='GPU index to use (default: 0)')
    args_to_check.append('gpu_idx')

    # add another optional flag to specify that we are using an older version of cisTEM
    # TODO: for now, just trying to catch the case where we use match_template not match_template_gpu, however,
    # there could be other cases where we need to be more specific if the input options change more over time.
    parser.add_argument('--old_cistem', action='store_true',
                        help='Use this flag if you are using an older version of cisTEM')
    args_to_check.append('old_cistem')

    # add an optional cpu flag
    parser.add_argument('--cpu', action='store_true',
                        help='Use this flag if you are using the cpu version of cisTEM')
    args_to_check.append('cpu')

    args = parser.parse_args()

    args.binary_name = wanted_binary_name
    args_to_check.append('binary_name')

    # currently no plan to have a gpu version
    args.results_binary_name = 'make_template_result'
    args_to_check.append('results_binary_name')

    # Check if we are using the cpu version or old version and if so modify the binary name with _gpu
    if not (args.old_cistem or args.cpu):
        args.binary_name += '_gpu'

    # Check if the binary exists
    if not os.path.isfile(os.path.join(args.binary_path, args.binary_name)):
        print('The binary ' + os.path.join(args.binary_path,
              args.binary_name) + ' does not exist')
        sys.exit(1)

    # Check if make_template_result binary exists
    if not os.path.isfile(os.path.join(args.binary_path, args.results_binary_name)):
        print('The binary ' + os.path.join(args.binary_path,
              args.results_binary_name) + ' does not exist')
        sys.exit(1)

    # if the optional data path is not given, use the default
    if args.test_data_path is None:
        args.test_data_path = default_data_dir

    # Check if the test data directory exists
    if not os.path.isdir(args.test_data_path):
        print('The test data directory [' +
              args.test_data_path + '] does not exist')
        print('Please provide a valid path to the test data directory as a second argument')
        sys.exit(1)

    # Check that the wanted output path exists and if not try to make it, if not error
    os.makedirs(args.output_file_prefix, exist_ok=True)

    args.args_to_check = args_to_check

    return args
