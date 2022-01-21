#!/usr/bin/env bash

set -e

# global variables
current_dir=$(pwd)
if [ -z "$1" ]; then
	config_dir='final'
else
	config_dir="$1"
fi

output_dir=$current_dir/output_1a

echo "Config directory: $config_dir"
echo "Output directory: $output_dir"

#######
# PRA #
#######
pra_dir='../pra/model'
pra_instance_dir=$pra_dir/model_instance

mkdir -p $pra_instance_dir/$config_dir
cp configuration/$config_dir/conf $pra_instance_dir/$config_dir/conf
cp configuration/$config_dir/config.sh $pra_instance_dir/$config_dir/config.sh
cd $pra_dir

./build_models.sh $config_dir
./predict.sh $config_dir
./evaluate_models.sh $config_dir

#######
# MLP #
#######
cd $current_dir

er_mlp_model_dir="../er_mlp/model"
er_mlp_instance_dir="$er_mlp_model_dir/model_instance"
er_mlp_log_dir="$current_dir/log/er_mlp_$config_dir.log"

mkdir -p "$er_mlp_instance_dir/$config_dir"
cp "configuration/$config_dir/er_mlp.ini" "$er_mlp_instance_dir/$config_dir/$config_dir.ini"
cd $er_mlp_model_dir

python3 -u build_network.py --dir $config_dir --logfile $er_mlp_log_dir --final_model
python3 -u predict.py --dir $config_dir --predict_file train_local.txt --logfile $er_mlp_log_dir --final_model
python3 -u predict.py --dir $config_dir --predict_file test.txt --logfile $er_mlp_log_dir --final_model

###########
# stacked #
###########
cd $current_dir

stacked_dir='../stacked'
stacked_instance_dir=$stacked_dir/model_instance

mkdir -p $stacked_instance_dir/$config_dir
cp 'configuration/'$config_dir'/stacked.ini' $stacked_instance_dir/$config_dir/$config_dir'.ini'
cd $stacked_dir

python3 -u build_model.py --dir $config_dir --er_mlp $config_dir --pra $config_dir --final_model
python3 -u evaluate.py --dir $config_dir --final_model

################
# post process #
################
cd $current_dir

data_path=`grep 'DATA_PATH' configuration/$config_dir/config.sh | cut -d "=" -f 2 | sed 's/"//g'`
hypotheses_file=$data_path/test.txt
confidence_file=$stacked_instance_dir/$config_dir/test/confidence_stacked.txt

echo "Hypotheses file: $hypotheses_file"
echo "Confidence file: $confidence_file"

mkdir -p $output_dir
cp $hypotheses_file $output_dir
cp $confidence_file $output_dir

hypotheses_file=$output_dir/test.txt
confidence_file=$output_dir/confidence_stacked.txt

sed  -i -E 's|#SEMICOLON#|:|g' $hypotheses_file
sed  -i -E 's|#SPACE#| |g' $hypotheses_file
sed  -i -E 's|#COMMA#|,|g' $hypotheses_file

awk -F"[][]" '{print $2}' $confidence_file > tmp && mv tmp $confidence_file

# join the hypotheses and their confidence and save to file
hypotheses_confidence_file=$output_dir/hypotheses_confidence.txt
echo "Saving hypotheses confidence file to: $hypotheses_confidence_file"

paste $hypotheses_file $confidence_file >> $hypotheses_confidence_file

# remove intermediate files
rm $hypotheses_file
rm $confidence_file
