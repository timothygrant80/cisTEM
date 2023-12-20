#!/usr/bin/env python3


import util.args as tmArgs
import util.make_tmp_runfile as mktmp
import util.run_job as runner

# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'


def main():

    args = tmArgs.parse_TM_args(wanted_binary_name)

    # 2 is the 6Q8Y_mature_60S.mrc template
    config = tmArgs.get_config(args, 'Yeast', 2, 0)

    # We want to do the full search (mostly) with defocus included, so override those defaults
    config['defocus_range'] = 900
    config['defocus_step'] = 300

    tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(config)
    elapsed_time = runner.run_job(tmp_filename_match_template)
    runner.run_job(tmp_filename_make_template_results)

    print('Time is : ' + str(elapsed_time))


# Check if main function and run
if __name__ == '__main__':
    main()
