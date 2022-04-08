#!/bin/bash

# This should be called from cistem_make.sh
n_threads=${1}
path_to_fix_deps=${2}

# Get all the objects we compiled that come from modules
tmp_objects=$(mktemp)
find . -name libmodules_a*.o > ${tmp_objects}


# # List of all the dep files that may include these modules.
# # If we don't limit the number of args (n) then only one proc (not P) is run
find . -name *.Po | xargs -n 3 -P $n_threads $path_to_fix_deps $tmp_objects

# Cleanup
rm $tmp_objects
