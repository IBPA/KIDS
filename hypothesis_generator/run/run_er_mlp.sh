#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# directories set
config_dir="$1"
current_dir=`pwd`
er_mlp_model_dir="../er_mlp/model"
er_mlp_instance_dir="$er_mlp_model_dir/model_instance"
er_mlp_log_dir="$current_dir/log/er_mlp_$config_dir.log"

# make the directory, copy .ini file there, and cd there
mkdir -p "$er_mlp_instance_dir/$config_dir"
cp "configuration/$config_dir/er_mlp.ini" "$er_mlp_instance_dir/$config_dir/$config_dir.ini"
cd $er_mlp_model_dir

# run the python scripts
python3 -u build_network.py --dir $config_dir --logfile $er_mlp_log_dir
python3 -u determine_thresholds.py --dir $config_dir --logfile $er_mlp_log_dir
python3 -u predict.py --dir $config_dir --predict_file train_local.txt --logfile $er_mlp_log_dir
python3 -u predict.py --dir $config_dir --predict_file dev.txt --logfile $er_mlp_log_dir
python3 -u predict.py --dir $config_dir --predict_file test.txt --logfile $er_mlp_log_dir
python3 -u evaluate_network.py --dir $config_dir --logfile $er_mlp_log_dir
