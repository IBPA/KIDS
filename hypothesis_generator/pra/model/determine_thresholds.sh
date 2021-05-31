#!/usr/bin/env bash

set -e

# directories
fold="$1"
current_dir=$(pwd)
prev_current_dir=$current_dir/..
model_instance_dir=$current_dir/model_instance
base_dir=$model_instance_dir/$fold
instance_dir=$base_dir/instance
io_util_dir='io_util'
pra_imp_dir='pra_imp'
dev_file='dev.txt'
dev_folder='dev'

# include files
. $base_dir/config.sh
. ./log.sh

log "using data from path '$DATA_PATH'"

log "changing directories to '$instance_dir'"
cd $instance_dir

# modify configurations
sed -i -e "s|given_negative_samples=true|given_negative_samples=false|g" conf
sed -i -e "s|blocked_field=THE_BLOCKED_FIELD|blocked_field=-1|g" conf
sed -i -e "s|target_relation=.*|target_relation=$start_relation|g" conf
sed -i -e "s|target_relation=THE_RELATION|target_relation=$start_relation|g" conf
sed -i -e "s|task=_TASK_|task=predict|g" conf
sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

# create folders
log "creating folders..."
mkdir -p $dev_folder
mkdir -p $dev_folder/queriesR_test
mkdir -p $dev_folder/queriesR_labels
mkdir -p $dev_folder/queriesR_tail
mkdir -p $dev_folder/predictions
mkdir -p $dev_folder/scores
mkdir -p $dev_folder/classifications
mkdir -p $dev_folder/thresholds
mkdir -p $dev_folder/thresholds_calibration

# modify configurations
sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf
sed -i -e "s|target_relation=$start_relation|target_relation=THE_RELATION|g" conf
sed -i -e "s|prediction_folder=.*/|prediction_folder=./$dev_folder/predictions/|g" conf
sed -i -e "s|test_samples=.*|test_samples=./$dev_folder/queriesR_test/<target_relation>|g" conf

while read p; do
	log "processing relation '$p'"
	sed -i -e "s|target_relation=THE_RELATION|target_relation=$p|g" conf

	log "creating test queries..."
	python3 $prev_current_dir/$io_util_dir/create_test_queries.py --data_file $DATA_PATH/$dev_file --predicate $p --dir $dev_folder

	grep -i -P "\t""$p""\t" $DATA_PATH/$dev_file | awk -F '\t' '{print"c$"$1 "\tc$" $3}' > ./$dev_folder/queriesR_tail/$p
	grep -i -P "\t""$p""\t" $DATA_PATH/$dev_file | awk -F '\t' '{print"c$"$1 "\tc$" $3 "\t" $4}' > ./$dev_folder/queriesR_labels/$p

	log "doing prediction..."
	java -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.LearnerPRA

	python3 $prev_current_dir/$io_util_dir/get_scores.py --predicate $p --dir $dev_folder
	python3 $prev_current_dir/$io_util_dir/determine_thresholds.py --predicate $p --dir $dev_folder

	sed -i -e "s|target_relation=$p|target_relation=THE_RELATION|g" conf
done <"selected_relations"

sed -i -e "s|task=predict|task=_TASK_|g" conf
sed -i -e "s|blocked_field=-1|blocked_field=THE_BLOCKED_FIELD|g" conf
