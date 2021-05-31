#!/usr/bin/env bash

set -e

pra_dir='../pra/model'
pra_instance_dir=$pra_dir/model_instance
config_dir="$1"
current_dir=$(pwd)

mkdir -p $pra_instance_dir/$config_dir
cp configuration/$config_dir/conf $pra_instance_dir/$config_dir/conf
cp configuration/$config_dir/config.sh $pra_instance_dir/$config_dir/config.sh
cd $pra_dir

./build_models.sh $config_dir
./determine_thresholds.sh $config_dir
./predict.sh $config_dir
./evaluate_models.sh $config_dir
