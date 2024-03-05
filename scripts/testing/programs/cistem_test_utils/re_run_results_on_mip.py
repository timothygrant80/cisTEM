import annoying_hack

from os import makedirs
from os import symlink
from os import remove
from os.path import join
import cistem_test_utils.make_tmp_runfile as mktmp
import cistem_test_utils.run_job as runner


def run_job(config):

    config['results_mip_to_use'] = 'mip.mrc'
    original_prefix = config['output_file_prefix']
    config['output_file_prefix'] = join(original_prefix, 'not_scaled')
    makedirs(config['output_file_prefix'], exist_ok=True)
    # The outputs from the search won't exist in this dir, so link them
    try:
        symlink(join(original_prefix, 'mip.mrc'), join(config['output_file_prefix'], 'mip.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'mip_scaled.mrc'), join(config['output_file_prefix'], 'mip_scaled.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'psi.mrc'), join(config['output_file_prefix'], 'psi.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'theta.mrc'), join(config['output_file_prefix'], 'theta.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'phi.mrc'), join(config['output_file_prefix'], 'phi.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'defocus.mrc'), join(config['output_file_prefix'], 'defocus.mrc'))
    except FileExistsError:
        pass
    try:
        symlink(join(original_prefix, 'pixel_size.mrc'), join(config['output_file_prefix'], 'pixel_size.mrc'))
    except FileExistsError:
        pass

    try:
        symlink(join(original_prefix, 'hist.txt'), join(config['output_file_prefix'], 'hist.txt'))
    except FileExistsError:
        pass

    tmp_filename_match_template, tmp_filename_make_template_results = mktmp.make_tmp_runfile(config)
    # WE won't re-run the search, so clean that up
    remove(tmp_filename_match_template)
    runner.run_job(tmp_filename_make_template_results)
