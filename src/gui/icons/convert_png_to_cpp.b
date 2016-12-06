# run as ./convert_png_to_cpp input.png output.cpp
optipng -o7 $1 
./bin2c -c $1 ${1%%.???}.cpp

