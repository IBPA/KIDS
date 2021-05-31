#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# HG root directory
hg_dir=`pwd`

# common variabls
config_dir="$hg_dir/run/configuration"
data_dir="$hg_dir/data"
default_data_dir="/path/to/data/directory"

# update paths in config.sh
echo "Updating filepaths in 'config.sh'..."
find "$config_dir" -type f -name "config.sh" | xargs sed -i 's|'$default_data_dir'|'$data_dir'|g'

# update paths in er_mlp.ini
echo "Updating filepaths in 'er_mlp.ini'..."
find "$config_dir" -type f -name "er_mlp.ini" | xargs sed -i 's|'$default_data_dir'|'$data_dir'|g'
