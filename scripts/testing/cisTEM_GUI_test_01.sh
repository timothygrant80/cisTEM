#!/bin/bash
#
# Basic cisTEM GUI test
#
# Alexis Rohou, March 2019
#
source $( dirname $(readlink -f "$0") )/cisTEM_xdotool.sh

cisTEM_path="/mnt/ext_home/work/git/cisTEM/build/Linux/debug_gnu/src"
export PATH=${cisTEM_path}:${PATH}


rm -fr /mnt/ext_home/work/MyNewProject
launch_cisTEM "${cisTEM_path}/cisTEM"
create_new_project "MyNewProject" "/mnt/ext_home/work"
click_on_assets_lhs_icon; sleep 1s
#click_on_actions_lhs_icon; sleep 1s
#click_on_results_lhs_icon; sleep 1s
#click_on_settings_lhs_icon; sleep 1s
#click_on_assets_movies; sleep 1s
#click_on_assets_images; sleep 1s
#click_on_assets_particlepositions; sleep 1s
#click_on_assets_3dvolumes; sleep 1s
#click_on_assets_refinementpackages; sleep 1s
#click_on_actions_alignmovies; sleep 1s
#click_on_actions_findctf; sleep 1s
#click_on_actions_findparticles; sleep 1s
#click_on_actions_2dclassify; sleep 1s
#click_on_actions_abinitio3d; sleep 1s
#click_on_actions_autorefine; sleep 1s
#click_on_actions_manualrefine; sleep 1s
#click_on_actions_generate3d; sleep 1s
#click_on_actions_sharpen3d; sleep 1s
#sleep 1s
import_run_profile "/mnt/ext_home/work/LaptopLocal_profile_nopath.txt"
sleep 2s
save_screenshot "screnshot_after_run_profile.png"
import_movies_no_gain_no_resample "/mnt/ext_home/work/160826_apo_test/movies/May08_03.2*.mrc" "300.0" "0.0" "1.5" "2.0"
sleep 5s
save_screenshot "screenshot_after_import_movies.png"
align_movies
sleep 1s
estimate_ctf
#close_project