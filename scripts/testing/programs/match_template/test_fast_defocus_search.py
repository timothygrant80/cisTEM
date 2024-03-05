#!/usr/bin/env python3

import annoying_hack

import cistem_test_utils.args as tmArgs
import cistem_test_utils.make_tmp_runfile as mktmp
import cistem_test_utils.run_job as runner

# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'


def main():

    args = tmArgs.parse_TM_args(wanted_binary_name)

    # 2 is the 6Q8Y_mature_60S.mrc template
    config = tmArgs.get_config(args, 'Yeast', 2, 0)

    # We want to do a defocus search to catch potential memory errors or bugs that only show up 
    # when the TM core object is used multiple times.
    config['defocus_range'] = 600
    config['defocus_step'] = 300

    # We are just looking for no crash, and not correctness, so use a fast search space.
    config.get('model')[config.get('ref_number')]['symmetry'] = 'O'
    # Override some default arguments to make this a fast search
    # We want it to take long enough to armortize any overhead, but not too long, ideally a minute or so
    config['out_of_plane_angle'] = 5.5
    config['in_plane_angle'] = 3.5

    tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(config)
    elapsed_time = runner.run_job(tmp_filename_match_template)
    runner.run_job(tmp_filename_make_template_results)

    print('Time is : ' + str(elapsed_time))


# Check if main function and run
if __name__ == '__main__':
    main()
