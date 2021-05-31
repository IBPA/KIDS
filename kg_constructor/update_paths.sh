#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# KG constructor root directory
kgc_dir=`pwd`

# common variabls
data_dir="$kgc_dir/data"

# update paths in data_path_file.txt
dataset_dir="$data_dir/dataset"
data_path_file="$data_dir/data_path_file.txt"
data_path_file_toy="$data_dir/data_path_file_toy.txt"
default_dataset_dir="/path/to/dataset/directory"

echo "Updating filepaths in '$data_path_file'..."
sed -i 's|'$default_dataset_dir'|'$dataset_dir'|g' $data_path_file

echo "Updating filepaths in '$data_path_file_toy'..."
sed -i 's|'$default_dataset_dir'|'$dataset_dir'|g' $data_path_file_toy

# update paths in the configuration files
config_dir="$kgc_dir/configuration"
output_dir="$kgc_dir/output"

create_config_file="$config_dir/create_kg_config.ini"
postprocess_config_file="$config_dir/postprocess_config.ini"
update_config_file="$config_dir/update_kg_config.ini"

default_data_dir="/path/to/data/directory"
default_output_dir="/path/to/output/directory"

echo "Updating filepaths in '$create_config_file'..."
sed -i 's|'$default_data_dir'|'$data_dir'|g' $create_config_file
sed -i 's|'$default_output_dir'|'$output_dir'|g' $create_config_file

echo "Updating filepaths in '$postprocess_config_file'..."
sed -i 's|'$default_data_dir'|'$data_dir'|g' $postprocess_config_file
sed -i 's|'$default_output_dir'|'$output_dir'|g' $postprocess_config_file

echo "Updating filepaths in '$update_config_file'..."
sed -i 's|'$default_data_dir'|'$data_dir'|g' $update_config_file
sed -i 's|'$default_output_dir'|'$output_dir'|g' $update_config_file
