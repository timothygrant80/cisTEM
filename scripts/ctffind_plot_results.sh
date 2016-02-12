#!/bin/bash 
#
#
# Plot the results from ctffind using gnuplot
#
# Alexis Rohou, June 2014
#
# Copyright 2014 Howard Hughes Medical Institute. All rights reserved.
# Use is subject to Janelia Farm Research Campus Software Copyright 1.1
# license terms ( http://license.janelia.org/license/jfrc_copyright_1_1.html )
#



# Parse arguments
if [ $# -ne 1 ]
then
	echo "Usage: `basename $0` output_from_ctffind_avrot.txt"
	exit 65
fi

input_fn=$1
output_fn=${1%%.???}.pdf
input_summary_fn=$(ls ${1%%_avrot.???}.${1##*.})

# Check whether gnuplot is available
if ! hash gnuplot 2>/dev/null; then
	echo "Gnuplot was not found on your system. Please make sure it is installed."
	exit -1
fi

# Check whether gnuplot >= 4.6 is available
gnuplot_version=$(  gnuplot --version | awk '{print $2}'  )
compare_result=$( echo "$gnuplot_version >= 4.6" | bc )
if [ $compare_result -le 0 ]; then
	echo "This script requires gnuplot version >= 4.6, but you have version $gnuplot_version"
	exit -1
fi

# Define a neat function
function transpose_ignore_comments {
gawk '
{
	if ($1 ~ /^#/) {
		print $0
	} else {
		for (i=1; i<=NF; i++)  {
       		a[NR,i] = $i
    	}
	}
}
NF>p { p = NF }
END {    
    for(j=1; j<=p; j++) {
        str=a[1,j]
        for(i=2; i<=NR; i++){
            str=str" "a[i,j];
        }
        print str
    }
}' $1
}

# CTFFind outputs data in lines, but gnuplot wants things in columns
transpose_ignore_comments $input_fn > /tmp/tmp.txt

# Let's grab useful values
pixel_size=$(gawk 'match($0,/Pixel size: ([0-9.]*)/,a) {print a[1]}' $input_fn)
number_of_micrographs=$(gawk 'match($0,/Number of micrographs: ([0-9]*)/,a) {print a[1]}' $input_fn)
lines_per_micrograph=6

# Let's grab values from the summary file
i=0
while read -a myArray
do
	if [[ ${myArray[0]} != \#* ]]; then
		df_one[++i]=${myArray[1]}
		df_two[i]=${myArray[2]}
		angast[i]=${myArray[3]}
		pshift[i]=${myArray[4]}
		score[i]=${myArray[5]}
		maxres=[i]=${myArray[6]}
	fi
done < $input_summary_fn

echo ${df_one[0]}

# Run Gnuplot
gnuplot > /dev/null 2>&1  <<EOF
#cat <<EOF > temp.txt
set border linewidth 1.5
if (strstrt(GPVAL_TERMINALS, 'pdfcairo') > 0) {
	set terminal pdfcairo size 10.0cm,4.0cm enhanced font 'Arial,9'
	set output '$output_fn'
} else {
	set terminal postscript size 10,4 enhanced font 'Arial,14'
	set output '${output_fn%%.pdf}.ps'
}

# color definitions
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 # --- blue
set style line 2 lc rgb 'red'     lt 1 lw 1 pt 7 fill transparent solid 0.5 # --- red
set style line 3 lc rgb 'orange'  lt 3 lw 1 pt 7 # --- orange
set style line 4 lc rgb 'light-blue' lt 1 lw 2 pt 7 # --- light blue
set style line 5 lc rgb 'green'   lt 1 lw 2 pt 7
set style line 6 lc rgb 'gray'    lt 1 lw 2 pt 7

set xlabel 'Spatial frequency(1/Å)'
set ylabel 'Amplitude (or cross-correlation)'
set yrange [-0.1:1.1]
set key outside

defocus_1_values="${df_one[*]}"
defocus_2_values="${df_two[*]}"
angast_values="${angast[*]}"
pshift_values="${pshift[*]}"
score_values="${score[*]}"
maxres_values="${maxres[*]}"

do for [current_micrograph=1:$number_of_micrographs] {
	def_1=sprintf('%.0f Å',word(defocus_1_values,current_micrograph)+0)
	def_2=sprintf('%.0f Å',word(defocus_2_values,current_micrograph)+0)
	angast=sprintf('%.1f °',word(angast_values,current_micrograph)+0)
	pshift=sprintf('%.2f rad',word(pshift_values,current_micrograph)+0)
	score=sprintf('%.3f',word(score_values,current_micrograph)+0)
	set title 'Micrograph '.current_micrograph." of $number_of_micrographs\n{/*0.5 Defocus 1: ".def_1.' | Defocus 2: '.def_2.' | Azimuth: '.angast.' | Phase shift: '.pshift.' | Score: '.score.'}'
	plot '/tmp/tmp.txt' using (\$1 / $pixel_size):(column(4+(current_micrograph-1)*$lines_per_micrograph)) w lines ls 3 title 'CTF fit', \
		 ''             using (\$1 / $pixel_size):(column(5+(current_micrograph-1)*$lines_per_micrograph)) w lines ls 1 title 'Quality of fit', \
	 	 ''             using (\$1 / $pixel_size):(column(3+(current_micrograph-1)*$lines_per_micrograph))  w lines ls 5 title 'Amplitude spectrum'
}
EOF

# Convert the postscript file to a pdf file if needed
if [ -f ${output_fn%%.pdf}.ps ]; then
	ps2pdf ${output_fn%%.pdf}.ps $output_fn
fi

# Fix the title of the PDF file
if hash pdftk 2>/dev/null; then
	echo "InfoKey: Title" > /tmp/pdftemp.txt
	echo "InfoValue: $1" >> /tmp/pdftemp.txt
	mv ${output_fn} /tmp/tmp.pdf
	pdftk /tmp/tmp.pdf update_info /tmp/pdftemp.txt output $output_fn
else
	echo "pdftk was not found on your system. Please install it to get improved metadata in your output PDF file."
fi

# Make a PNG version
#convert -density 300 -flatten ${output_fn} ${output_fn%%.pdf}.png

# Let the user know where to find results
echo " "
echo "Generated plots of 1D profiles and fits: ${output_fn}"
echo " "
