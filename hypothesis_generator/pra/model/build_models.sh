#!/usr/bin/env bash

set -e

# functions
containsElement () {
	local e match="$1" ret="false"
	shift
	for e;
	do
		if [[ "$e" == "$match" ]]; then
			ret="true"
			break
		fi
	done
	echo "$ret"
}

# directories
fold="$1"
current_dir=$(pwd)
prev_current_dir=$current_dir/..
model_instance_dir=$current_dir/model_instance
base_dir=$model_instance_dir/$fold
io_util_dir='io_util'
pra_imp_dir='pra_imp'
data_handler_dir='data_handler'
train_folder='train'

# include files
. $base_dir/config.sh
. ./log.sh

log "using data from path '$DATA_PATH'"

# move into model instance directory
cd $model_instance_dir

# generate instance directory
instance_dir=$base_dir/instance

if [ -d "$instance_dir" ]; then
  rm -rfd $instance_dir
fi
mkdir $instance_dir

# copy conf file into the instance directory
log "copying configuration..."
cp $base_dir/conf $instance_dir

# replace some parts of the conf file
cd $instance_dir
sed -i -e "s|blocked_field=THE_BLOCKED_FIELD|blocked_field=0|g" conf
sed -i -e "s|target_relation=.*|target_relation=$start_relation|g" conf
sed -i -e "s|target_relation=THE_RELATION|target_relation=$start_relation|g" conf
sed -i -e "s|task=_TASK_|task=train|g" conf

# process data
log "processing data..."
if  [ "$use_domain" == "true" ]; then
	python3 $prev_current_dir/$data_handler_dir/pra_data_processor.py --data_path $DATA_PATH --train_file $train_file --use_domain
else
	python3 $prev_current_dir/$data_handler_dir/pra_data_processor.py --data_path $DATA_PATH --train_file $train_file
fi

# produce edges file
log "producing edges file..."
java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.data.WKnowledge createEdgeFile $instance_dir/ecoli_generalizations.csv 0.1 edges

# create graphs folder
mkdir -p graphs
mkdir -p graphs/pos

sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

# move edges file into the graphs folder
mv ecoli_generalizations.csv.p0.1.edges graphs/pos/.

# index edge file to a compact graph reprensentation
log "indexing edge file to a compact graph reprensentation..."
java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.SmallJobs indexGraph ./graphs/pos edges

# create train folder
mkdir -p $train_folder
mkdir -p $train_folder/queriesR_train

# create queries
log "creating positive queries..."
java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.SmallJobs createQueries ./graphs/pos $train_folder/queriesR_train/ true false

# if we are using negatives
if [ -f "$instance_dir/"ecoli_generalizations_neg.csv ] && [ "$use_negatives" == true ]; then
	log "creating negative queries..."
	sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf

	# produce edges file
	java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.data.WKnowledge createEdgeFile $instance_dir/ecoli_generalizations_neg.csv 0.1 edges

	mkdir graphs/neg

	mv ecoli_generalizations_neg.csv.p0.1.edges graphs/neg/.

	sed -i -e "s|/graphs/pos|/graphs/neg|g" conf

	# index edge file to a compact graph reprensentation
	java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.SmallJobs indexGraph ./graphs/neg edges

	mkdir $train_folder/queriesR_train_neg

	# create queries
	java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.SmallJobs createQueries ./graphs/neg $train_folder/queriesR_train_neg/ true false

	sed -i -e "s|/graphs/neg|/graphs/pos|g" conf

	python3 $prev_current_dir/$io_util_dir/merge_queries.py --dir $train_folder
else
	sed -i -e "s|given_negative_samples=true|given_negative_samples=false|g" conf
fi

sed -i -e "s|pra_neg_mode_v4.jar|$prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar|g" conf

# train the models
mkdir models

sed -i -e "s|target_relation=$start_relation|target_relation=THE_RELATION|g" conf
while read p; do
	sed -i -e "s|target_relation=THE_RELATION|target_relation=$p|g" conf

	does_not_have_negatives=$(containsElement $p "${no_negatives[@]}")
	log "processing relation '$p'"

	log "does_not_have_negatives: $does_not_have_negatives"

	if  [ "$does_not_have_negatives" == true ] || [ "$use_negatives" != true ]; then
		sed -i -e "s|given_negative_samples=true|given_negative_samples=false|g" conf
	else
		sed -i -e "s|given_negative_samples=false|given_negative_samples=true|g" conf
	fi

	# train model (or do predictions) for a relation with parameters
	log "training models..."
	java -Xms6G -Xmx6G -cp $prev_current_dir/$pra_imp_dir/pra_neg_mode_v4.jar edu.cmu.pra.LearnerPRA

	model=$(find `pwd pos/$p` -name train.model)
	mv $model models/$p
	rm -rfd pos/$p

	sed -i -e "s|target_relation=$p|target_relation=THE_RELATION|g" conf

done <"selected_relations"

sed -i -e "s|task=train|task=_TASK_|g" conf
sed -i -e "s|blocked_field=0|blocked_field=THE_BLOCKED_FIELD|g" conf
