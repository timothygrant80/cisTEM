#!/usr/bin/env python3

import annoying_hack

from os.path import join as join
from os import makedirs
import cistem_test_utils.args as tmArgs
import cistem_test_utils.make_tmp_runfile as mktmp
import cistem_test_utils.run_job as runner
import cistem_test_utils.re_run_results_on_mip as re_runner

import sys
# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'


def test_val(args):
    args.output_file_prefix = '/shit'
    print("inside", args.output_file_prefix)


def main():

    args = tmArgs.parse_TM_args(wanted_binary_name)

    elapsed_time = [0, 0, 0, 0]
    for image_number in range(0, 4):

        config = tmArgs.get_config(args, 'Kras', 0, image_number)

        tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(config)

        elapsed_time[image_number] = runner.run_job(tmp_filename_match_template)
        runner.run_job(tmp_filename_make_template_results)

        # Now run the results again, overriding the default scaled mip and use the mip
        re_runner.run_job(config)

    for image_number in range(0, 4):
        print('Time is : ' + str(elapsed_time[image_number]))


# Check if main function and run
if __name__ == '__main__':
    main()
