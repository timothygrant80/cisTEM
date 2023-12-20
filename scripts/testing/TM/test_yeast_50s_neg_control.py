#!/usr/bin/env python3


from os.path import join as join
import util.args as tmArgs
import util.make_tmp_runfile as mktmp
import util.run_job as runner
import util.re_run_results_on_mip as re_runner

# By default the "_gpu" suffix will be added unless the --old-cistem flag is used
# or the --cpu flag is used
wanted_binary_name = 'match_template'


def main():

    args = tmArgs.parse_TM_args(wanted_binary_name)

    # 2 is the 6Q8Y_mature_60S.mrc template
    # 3 is neg_control_50S_7bv8.mrc
    config = tmArgs.get_config(args, 'Yeast', 3, 0)

    tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(config)
    elapsed_time = runner.run_job(tmp_filename_match_template)
    runner.run_job(tmp_filename_make_template_results)

    re_runner.run_job(config)

    print('Time is : ' + str(elapsed_time))


# Check if main function and run
if __name__ == '__main__':
    main()
