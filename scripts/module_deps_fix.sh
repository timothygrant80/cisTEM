#!/bin/bash

# This should only be called from module_deps_find.sh and is run in paralle.


module_objects=${1}
Po_name=${2}

# Search all files to indicate they depend on a given set of modules
a=($(awk '{if(/.c\+\+m/ && !/^CXX_IMPORTS/) print $0}' $Po_name))

record_length=${#a[@]}
my_tmp=$(tempfile)
if [ $record_length -gt 0 ] ; then
    # The first word is just the path to the target object file, this stays unchanged
    modified_dependency="${a[0]}"
    for word in $(seq 1 $((record_length - 1))) ; do
        module_name=$(basename ${a[$word]} .c++m)
        dep_src=$(awk -v MN=$module_name -F "libmodules_a-" '{if($2==MN".o") print $1MN".cpp"}' $module_objects)
        modified_dependency="$modified_dependency ${dep_src}"
    done
    # Now print all but that line out to a temp file
    awk '{if(/.c\+\+m/ && !/^CXX_IMPORTS/) ; else print $0}' $Po_name > $my_tmp
    echo "$modified_dependency" >> $my_tmp
    mv $my_tmp $Po_name
else    
    rm -f $my_tmp
fi

