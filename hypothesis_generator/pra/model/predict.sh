#!/usr/bin/env bash

set -e

fold="$1"
current_dir=$(pwd)
prev_current_dir=$current_dir/..
model_instance_dir=$current_dir/model_instance
base_dir=$model_instance_dir/$fold
instance_dir=$base_dir/instance
io_util_dir='io_util'
pra_imp_dir='pra_imp'
test_file='data.txt'

# include files
. $base_dir/config.sh
. ./log.sh

prediction_folder=${predict_file%.txt}
log "prediction folder path: '$prediction_folder'"

log "using data from path '$DATA_PATH'"

log "changing directories to '$instance_dir'"
cd $instance_dir

# modify configurations
sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf
sed -i -e "s|blocked_field=THE_BLOCKED_FIELD|blocked_field=-1|g" conf
sed -i -e "s|target_relation=.*|target_relation=$start_relation|g" conf
sed -i -e "s|target_relation=THE_RELATION|target_relation=$start_relation|g" conf
sed -i -e "s|task=_TASK_|task=predict|g" conf
sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

# create folders
log "creating folders..."
mkdir -p $prediction_folder
mkdir -p $prediction_folder/queriesR_test
mkdir -p $prediction_folder/queriesR_labels
mkdir -p $prediction_folder/queriesR_tail
mkdir -p $prediction_folder/predictions
mkdir -p $prediction_folder/scores
mkdir -p $prediction_folder/classifications

# copy train_local.txt
log "copying '$DATA_PATH/$predict_file' into '$prediction_folder/data.txt'"
cp $DATA_PATH/$predict_file $prediction_folder/data.txt

# modify configurations
sed -i -e "s|prediction_folder=.*/|prediction_folder=$prediction_folder/predictions/|g" conf
sed -i -e "s|test_samples=.*|test_samples=$prediction_folder/queriesR_test/<target_relation>|g" conf
sed -i -e "s|target_relation=$start_relation|target_relation=THE_RELATION|g" conf

while read p; do
	log "processing relation '$p'"
	sed -i -e "s|target_relation=THE_RELATION|target_relation=$p|g" conf

	log "creating test queries..."
	python3 $prev_current_dir/$io_util_dir/create_test_queries.py --data_file $prediction_folder/$test_file --predicate $p --dir $prediction_folder

	grep -i -P "\t""$p""\t" $prediction_folder/$test_file | awk -F '\t'  '{print"c$"$1 "\tc$" $3}' > ./$prediction_folder/queriesR_tail/$p
	grep -i -P "\t""$p""\t" $prediction_folder/$test_file | awk -F '\t'  '{print"c$"$1 "\tc$" $3 "\t" $4}' > ./$prediction_folder/queriesR_labels/$p

	log "doing prediction..."
	java -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.LearnerPRA

	python3 $prev_current_dir/$io_util_dir/get_scores.py --predicate $p --dir $prediction_folder

	if [ "$fold" != "final" ]; then
		python3 $prev_current_dir/$io_util_dir/classify.py --predicate $p --dir $prediction_folder
	fi

	sed -i -e "s|target_relation=$p|target_relation=THE_RELATION|g" conf
done <"selected_relations"

log "performing evaluation..."
if [ "$fold" != "final" ]; then
	python3 $prev_current_dir/$io_util_dir/evaluate.py --dir $prediction_folder
else
	python3 $prev_current_dir/$io_util_dir/evaluate.py --dir $prediction_folder --final_model
fi

sed -i -e "s|task=predict|task=_TASK_|g" conf
sed -i -e "s|blocked_field=-1|blocked_field=THE_BLOCKED_FIELD|g" conf
