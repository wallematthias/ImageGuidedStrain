#!/bin/bash

# Check if two arguments were passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <fixed> <moved>"
    exit 1
fi

# Assign the command line arguments to variables
fixed=$1
moved=$2

# Define the output file name
config_file="${moved}.cfg"

# Generate the config file
echo "fixed = ${fixed}.dcm" > $config_file
echo "moved = ${moved}.dcm" >> $config_file
echo "nodespacing = 24" >> $config_file
echo "with_memory = False" >> $config_file
echo "registered=${moved}-registered.xdmf" >> $config_file
echo "map=${moved}.xdmf" >> $config_file
echo "save_intermediate_frames=False" >> $config_file
echo "lambda = auto" >> $config_file
echo "max_iterations = 100" >> $config_file

echo "Config file ${config_file} generated successfully."