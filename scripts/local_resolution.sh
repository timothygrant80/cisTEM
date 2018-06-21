#!/bin/bash
#
# Parallelize local resolution estimation so it can run in a reasonable time.
# Make sure the program "local_resolution" is in your PATH before running
#
# Alexis Rohou, May-June 2018
#

#
# User-supplied parameters
#
input_volume_fsc_one=$1
input_volume_fsc_two=$2
input_volume_mask=$3
pixel_size=$4
box_size=$5
sampling_step=$6
slices_per_processor=$7
total_number_of_slices=$8
output_volume=$9

if [ "$#" -ne 9 ]; then
	echo " "
	echo "9 arguments are expected, you only supplied $#"
	echo "Usage: $0 input_fsc_vol_one input_fsc_vol_two input_mask_vol pixel_size box_size sampling_step slices_per_processor total_number_of_slices output_volume"
	echo "Suggested defaults:"
	echo "box_size = 20"
	echo "sampling_step = 2 or 4"
	echo " "
	exit
fi

#
# Loop over processors
#
first_slice=1
processor_counter=1
job_pid=()
while [ $first_slice -le $total_number_of_slices ]
do
	last_slice=$( echo "$first_slice + $slices_per_processor - 1" | bc)
	if [ $last_slice -gt $total_number_of_slices ]; then last_slice=$total_number_of_slices; fi
	log_fn="local_resolution_${processor_counter}.log"

	echo "$first_slice $last_slice $log_fn"

#
local_resolution<<eof > $log_fn 2>&1 &
$input_volume_fsc_one
$input_volume_fsc_two
$input_volume_mask
tmp_${processor_counter}.mrc
$pixel_size
$first_slice
$last_slice
$sampling_step
$box_size
eof
	
	# Remember the PID for this job
	job_pid+=($!)
	
	# Get ready for next increment
	(( first_slice = $last_slice + 1 ))
	(( processor_counter = $processor_counter + 1 ))
done

#
# Wait for the local resolution jobs to complete
#
echo "Waiting for local resolution estimation jobs to complete..."
for pid in ${job_pid[*]}; do
	wait $pid
	echo "Job $pid finished..."
done

#
# Merge the intermediate volumes
# 
log_fn="local_resolution_finalize.log"
local_resolution_finalize<<eof > $log_fn 2>&1
tmp_1.mrc
$slices_per_processor
$sampling_step
$output_volume
eof