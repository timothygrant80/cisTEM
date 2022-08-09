#!/bin/bash

# This must be run from the root of the repository.
if [[ ! -f .clang-format ]] ; then 
    echo "Did not find the .clang-format file. Are you in the root of the repository?"
    exit 1
fi

# Run clang format with the "fixed" style defined in .clang-format

# To ensure the correct version and options are specified, we use the hash of a pre/post converted test file using md5sum
original_format_hash=440fdc47889766cbf5c6036feb8c4934
post_format_hash=88ad6d70d449ba758ea127b2074256f5

# Requires clang-format-14 
clang_format_verision=14
clang_binary=""
# Check version of clang-format, first looking for the generic name, override by specific name (or use if generic not available.)
found_clang_version=0
for this_binary in clang-format clang-format-${clang_format_verision} ; do
    # First check that the binary is there.
    type $this_binary >/dev/null 2>&1 && check=$($this_binary --version | grep -e "clang-format version ${clang_format_verision}.")
    # If it is, check that it is the right version.
    if [[ $? -eq 0 ]]; then
        found_clang_version=1
        clang_binary=$this_binary
        break
    fi
done


if [[ $found_clang_version -eq 0 ]]; then
    echo "clang-format version ${clang_format_verision} not found"
    exit 1
fi
# first argumetn is either:
# all: format everything in your src directory
# /full/path/to/file.cpp:

format_all=0
test_output=0
# second argument is optionally "test" where your output will be placed in /tmp for inspection
if [[ $# -eq 1 || $# -eq 2 ]] ; then

    if [[ $1 == "console_test" ]]; then
        # First we check the input file is the correct ground truth.
        test_hash_input=$(md5sum testing/pre_format_test.cpp | cut -d' ' -f1)
        if [[ $test_hash_input != $original_format_hash ]]; then
            echo "Input file is not the correct ground truth"
            exit 1
        fi
        # Now we format the file and check the output is correct.
        $clang_binary -style=file testing/pre_format_test.cpp > testing/post_format_test.cpp
        
        test_hash_output=$(md5sum testing/post_format_test.cpp | cut -d' ' -f1)
        if [[ $test_hash_output != $post_format_hash ]]; then
            echo "Output file is not the correct ground truth"
            exit 1
        else
            exit 0
        fi
    fi
    
    if [[ $1 == "all" ]] ; then
        format_all=1
    fi
    if [[ $2 == "test" ]] ; then
        test_output=1
    fi
else
    echo 'One or two arguments are required'
    exit 1
fi

file_path=""
tmp_file="./format_tmp"
if [[ $format_all -eq 1 ]] ; then
    echo "Formatting all files in src directory"
    find ./src -name '*.cpp' -o -name '*.cu' -o -name '*.h'  | grep -v icons | grep -v ProjectX > $tmp_file
else
    echo "Formatting $1"
    echo $1 > $tmp_file
fi

cat $tmp_file | 
    while read file_name ; do
        if [[ $test_output -eq 1 ]] ; then
            file_path="/tmp"
        else
            file_path=$(dirname $file_name)
        fi
        cp $file_name $tmp_file.cpp
        $clang_binary -style=file ${tmp_file}.cpp > ${file_path}/$(basename $file_name)
    done
rm $tmp_file
rm $tmp_file.cpp