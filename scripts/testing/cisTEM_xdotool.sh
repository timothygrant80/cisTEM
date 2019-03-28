#!/bin/bash 
#
# GUI testing for cisTEM, using xdotool. This file is essentially a "library"
# of functions.
#
# This is intended to be run within a Docker / virtual machine,
# but should be OK to run in an actual Linux environment.
#
# Alexis Rohou, March 2019
#
# Note: a more powerful alternative might be to use cnee or Xnee, which
# makes it possible to monitor X events and synchronize the tests to them
# (e.g. wait for a popup window to appear, then perform next action), but 
# the learning curve seems steeper.
#
# On MacOS (Xquartz), might have to run
#
#

# cisTEM window size
cisTEM_window_size_x=1366 
cisTEM_window_size_y=768

# Directory where the script lives
script_dir="$( dirname $(readlink -f "$0") )"

# Filenames for GUI templates
template_fn_finish="${script_dir}/template_finish.png"

# Let's define the coordinates of the menu items
project_menu_item_x=30
project_menu_item_y=40
new_project_menu_item_x=$((   project_menu_item_x + 0 ))
new_project_menu_item_y=$((   project_menu_item_y + 20 ))
close_project_menu_item_x=$(( project_menu_item_x + 0 ))
close_project_menu_item_y=$(( project_menu_item_y + 100 ))

# A safe place to click when we want to reactivate the window
safe_place_to_click_x=$(( project_menu_item_x + 200 ))
safe_place_to_click_y=$(( project_menu_item_y + 0 ))

# Icons
overview_lhs_icon_x=55
overview_lhs_icon_y=110
assets_lhs_icon_x=$(( overview_lhs_icon_x + 0 ))
assets_lhs_icon_y=$(( overview_lhs_icon_y + 115 ))
actions_lhs_icon_x=$(( overview_lhs_icon_x + 0 ))
actions_lhs_icon_y=$(( overview_lhs_icon_y + 225 ))
results_lhs_icon_x=$(( overview_lhs_icon_x + 0 ))
results_lhs_icon_y=$(( overview_lhs_icon_y + 335 ))
settings_lhs_icon_x=$(( overview_lhs_icon_x + 0 ))
settings_lhs_icon_y=$(( overview_lhs_icon_y + 445 ))
top_icon_1_x=$(( overview_lhs_icon_x + 105 ))
top_icon_2_x=$(( top_icon_1_x + 90 ))
top_icon_3_x=$(( top_icon_2_x + 90 ))
top_icon_4_x=$(( top_icon_3_x + 90 ))
top_icon_5_x=$(( top_icon_4_x + 90 ))
top_icon_6_x=$(( top_icon_5_x + 90 ))
top_icon_7_x=$(( top_icon_6_x + 90 ))
top_icon_8_x=$(( top_icon_7_x + 90 ))
top_icon_9_x=$(( top_icon_8_x + 90 ))

#
# POINT AND CLICK FUNCTIONS
#
click_on_xy() {
	xdotool mousemove --sync "$1" "$2" click 1
}
# Menu
click_on_project() { 
	click_on_xy ${project_menu_item_x} ${project_menu_item_y}
}
click_on_new_project() { 
	click_on_xy ${new_project_menu_item_x} ${new_project_menu_item_y}
}
click_on_close_project() { 
	click_on_xy ${close_project_menu_item_x} ${close_project_menu_item_y}
}
# LHS icons
click_on_assets_lhs_icon() {
	click_on_xy ${assets_lhs_icon_x} ${assets_lhs_icon_y}
}
click_on_actions_lhs_icon() {
	click_on_xy ${actions_lhs_icon_x} ${actions_lhs_icon_y}
}
click_on_results_lhs_icon() {
	click_on_xy ${results_lhs_icon_x} ${results_lhs_icon_y}
}
click_on_settings_lhs_icon() {
	click_on_xy ${settings_lhs_icon_x} ${settings_lhs_icon_y}
}
# Top icons
click_on_top_icon_1() {
	click_on_xy ${top_icon_1_x} ${overview_lhs_icon_y}
}
click_on_top_icon_2() {
	click_on_xy ${top_icon_2_x} ${overview_lhs_icon_y}
}
click_on_top_icon_3() {
	click_on_xy ${top_icon_3_x} ${overview_lhs_icon_y}
}
click_on_top_icon_4() {
	click_on_xy ${top_icon_4_x} ${overview_lhs_icon_y}
}
click_on_top_icon_5() {
	click_on_xy ${top_icon_5_x} ${overview_lhs_icon_y}
}
click_on_top_icon_6() {
	click_on_xy ${top_icon_6_x} ${overview_lhs_icon_y}
}
click_on_top_icon_7() {
	click_on_xy ${top_icon_7_x} ${overview_lhs_icon_y}
}
click_on_top_icon_8() {
	click_on_xy ${top_icon_8_x} ${overview_lhs_icon_y}
}
click_on_top_icon_9() {
	click_on_xy ${top_icon_9_x} ${overview_lhs_icon_y}
}
click_on_assets_movies() {
	click_on_assets_lhs_icon; sleep 1s
	click_on_top_icon_1
}
click_on_assets_images() {
	click_on_assets_lhs_icon; sleep 1s
	click_on_top_icon_2
}
click_on_assets_particlepositions() {
	click_on_assets_lhs_icon; sleep 1s
	click_on_top_icon_3
}
click_on_assets_3dvolumes() {
	click_on_assets_lhs_icon; sleep 1s
	click_on_top_icon_4
}
click_on_assets_refinementpackages() {
	click_on_assets_lhs_icon; sleep 1s
	click_on_top_icon_5
}
click_on_actions_alignmovies() {
	click_on_actions_lhs_icon; sleep 1s
	click_on_top_icon_1
}
click_on_actions_findctf() {
	click_on_actions_lhs_icon; sleep 1s
	click_on_top_icon_2
}
click_on_actions_findparticles() {
	click_on_actions_lhs_icon; sleep 1s
	click_on_top_icon_3
}
click_on_actions_2dclassify() {
	click_on_actions_lhs_icon; sleep 1s
	click_on_top_icon_4
}
click_on_actions_abinitio3d() {
	click_on_actions_lhs_icon; sleep 1s
	click_on_top_icon_5
}
click_on_actions_autorefine() {
	click_on_actions_lhs_icon; sleep 1s
	click_on_top_icon_6
}
click_on_actions_manualrefine() {
	click_on_actions_lhs_icon; sleep 1s
	click_on_top_icon_7
}
click_on_actions_generate3d() {
	click_on_actions_lhs_icon; sleep 1s
	click_on_top_icon_8
}
click_on_actions_sharpen3d() {
	click_on_actions_lhs_icon; sleep 1s
	click_on_top_icon_9
}
click_on_start_job_button() {
	click_on_xy 850 746
	sleep 0.2s
}
click_on_finish_button() {
	click_on_xy 1304 743
	sleep 0.2s
}


# This is necessary because the main cisTEM window changes
# ID everytime a popup is shown. I'm not sure why this is.
update_WID() {
	WID=$(xdotool search --name --onlyvisible "cisTEM" | tail -1)
}

activate_cisTEM_window() {
	click_on_xy "${safe_place_to_click_x}" "${safe_place_to_click_y}"
	sleep 0.2s
	click_on_xy "${safe_place_to_click_x}" "${safe_place_to_click_y}"
	update_WID
}


# Launch cisTEM
launch_cisTEM() {
	if [ $# -ne 1 ]; then echo "Need one argument"; fi
	$1 > cisTEM.log 2>&1 &
	sleep 5s
	# Get the window ID
	update_WID
	echo "Launched cisTEM. Window ID = ${WID}"
	# Size
	xdotool windowsize "$WID" "$cisTEM_window_size_x" "$cisTEM_window_size_y"
	# Position
	xdotool windowmove --sync "$WID" 0 0
	# Activate the window
	#xdotool windowactivate --sync "${WID}"
	activate_cisTEM_window
}




# Create a new project
create_new_project() {
	if [ $# -ne 2 ]; then echo "Need two arguments"; fi
	# Click on Project
	click_on_project
	sleep 1

	# Click on New Project
	click_on_new_project
	sleep 1

	# Make sure the Create New Project wizard is selected
	wizardWID=$(xdotool search --name "Create New Project"  | tail -1)
	xdotool windowraise "${wizardWID}"
	xdotool type "$1"
	xdotool key Tab
	xdotool type "$2"
	xdotool key Return
	sleep 1s
	activate_cisTEM_window
}

# Import a run profile text file
import_run_profile() {
	if [ $# -ne 1 ]; then echo "Need one argument"; fi
	click_on_settings_lhs_icon; sleep 1s
	click_on_xy 435 274 # the remove button
	sleep 1s
	click_on_xy 435 365 # the import button
	sleep 1s
	xdotool type "$1"
	sleep 1s
	xdotool key Return
	sleep 1s
	activate_cisTEM_window
}

# Import movies
import_movies_no_gain_no_resample() {
	if [ $# -ne 5 ]; then echo "Need 5 arguments"; fi
	click_on_assets_movies
	sleep 1s
	click_on_xy 578 635
	sleep 1s
	click_on_xy 551 305 # Add files
	sleep 1s
	xdotool type "$1"
	sleep 1s
	xdotool key Return
	sleep 1s # This should probably be a longer sleep?
	# We are leaving the file browser with the flename type box active
	xdotool key --delay 500 Tab Tab Tab
	sleep 1s
	xdotool type "$2" # voltage
	sleep 1s
	xdotool key Tab Tab
	xdotool type "$3" # Cs
	sleep 1s
	xdotool key Tab
	xdotool type "$4" # psize
	sleep 1s
	xdotool key Tab
	xdotool type "$5"
	sleep 1s
	xdotool key --delay 300 Tab Tab Tab Tab Tab Tab
	xdotool key Return
	sleep 2s
	activate_cisTEM_window
}


# Close the current project
close_project() {
 	click_on_project
 	sleep 1
 	click_on_close_project
}

# Take a screenshot of the cisTEM window and save it to the supplied
# filename
save_screenshot() {
	if [ $# -ne 1 ]; then echo "Need one argument"; fi
	xwd -id "${WID}" | convert xwd:- "$1"
}

# Take a screenshot of the bottom right corner
# Supply x y dimensions and filename
# To capture the Finish button, 120x65 should be enough
save_cropped_screenshot_bottom_right_corner() {
	if [ $# -ne 3 ]; then echo "Need 3 arguments"; fi
	# Let's compute the top left (tl) coordinates of the rectangle we want to cut out
	tlx=$(( $cisTEM_window_size_x - $1 + 1 ))
	tly=$(( $cisTEM_window_size_y - $2 + 1 ))
	xwd -id "${WID}" | convert -crop "${1}x${2}+${tlx}+${tly}" +repage xwd:- "$3"
}

# Is the finish button showing in the bottom right corner?
# Returns 0 if the finish button was found, 1 otherwise
is_finish_button_showing() {
	tmp_screenshot_fn="tmp_screenshot.png"
	save_cropped_screenshot_bottom_right_corner 140 85 "${tmp_screenshot_fn}"
	compare -metric RMSE -subimage-search "${tmp_screenshot_fn}" "${template_fn_finish}" null: > /dev/null 2>&1
}

# Wait for the finish button to show
wait_for_finish_button() {
	sleep 1s
	activate_cisTEM_window
	sp="/-\|"
	echo -n 'Waiting for finish button to show...'
	until is_finish_button_showing
	do
		printf "\b${sp:i++%${#sp}:1}"
		sleep 5s
	done
	echo " "
	echo "Finish button detected. Proceeding."
}

align_movies() {
	click_on_actions_alignmovies
	sleep 1s
	click_on_start_job_button
	sleep 10s
	wait_for_finish_button
	click_on_finish_button
}

estimate_ctf() {
	click_on_actions_findctf
	sleep 1s
	click_on_start_job_button
	sleep 10s
	wait_for_finish_button
	click_on_finish_button
}

