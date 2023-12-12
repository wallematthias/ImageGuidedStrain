#!/bin/bash

# Find all files ending with *percent*.mha and store them in an array
files=(*percent*.mha)

# Sort the array of files by filename
IFS=$'\n' sorted_files=($(sort <<<"${files[*]}"))
unset IFS

# Get the total number of files
total_files=${#sorted_files[@]}

# Run igstrain command for each file with the first file as file[0]
for ((i = 1; i < total_files; i++)); do
    file_0=${sorted_files[0]}
    file_i=${sorted_files[$i]}

    # Run your igstrain command here using file_0 and file_i
    echo "igstrain $file_0 $file_i"
    igstrain $file_0 $file_i
done