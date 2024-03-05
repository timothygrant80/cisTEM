#!/usr/bin/env python3

import annoying_hack

from os.path import join as join
import cistem_test_utils.args as tmArgs
import cistem_test_utils.make_tmp_runfile as mktmp
import cistem_test_utils.run_job as runner
import cistem_test_utils.re_run_results_on_mip as re_runner
import sys

# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'


def main():

    args = tmArgs.parse_TM_args(wanted_binary_name)

    # Try first with a single plane
    # We want to do the full search (mostly) with defocus included, so override those defaults
    # args.defocus_range = 900
    # args.defocus_step = 300

    # 0 is the fring image
    # 1 is the all fun stuff image
    config = tmArgs.get_config(args, 'Lamella_from_je', 1, 1)

    tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(config)
    elapsed_time = runner.run_job(tmp_filename_match_template)
    runner.run_job(tmp_filename_make_template_results)

    # Now run the results again, overriding the default scaled mip and use the mip
    re_runner.run_job(config)
    print('Time is : ' + str(elapsed_time))


# Check if main function and run
if __name__ == '__main__':
    main()
