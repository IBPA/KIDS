#!/bin/bash
set -e

cd ./src/OpenKE

# hyperparameter search
python transe_kids.py \
	--dataset=ecoli \
	--mode=evaluate \
	--based_on=f1 \
	--random_state=530 \
	--batch_size=512 \
	--neg_ent=25 \
	--dim=1024 \
	--p_norm=1,2 \
	--margin=6 \
	--adv_temperature=1 \
	--alpha=0.001 \
	--train_times=1000 \
	--output_filename=hyperparameter_search_results.pkl
