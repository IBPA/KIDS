#!/usr/bin/env bash

set -e

stacked_dir='../stacked/'
stacked_instance_dir=$stacked_dir/model_instance/
config_dir="$1"
current_dir=$(pwd)

mkdir -p $stacked_instance_dir'/'$config_dir
cp 'configuration/'$config_dir'/stacked.ini' $stacked_instance_dir'/'$config_dir'/'$config_dir'.ini'
cd $stacked_dir
python3 -u build_model.py --dir $config_dir --er_mlp $config_dir --pra $config_dir
python3 -u evaluate.py --dir $config_dir
