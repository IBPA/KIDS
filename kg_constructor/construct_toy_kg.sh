#!/bin/bash
set -e

cd ./src

# Step 1. Create the intermediate knowledge graph.
python create_intermediate_kg.py \
	--dataset=toy \
	--skip_inference \
	--skip_remove \
	--n_workers=1

# Step 2. Check and validate the inconsistencies.

#    Put the validation results into the validation
#    folder (e.g. ./data/toy/validation).

# Step 3. Create the inconsistency-free knowledge graph.
python create_inconsistency_free_kg.py --dataset=toy

# Step 4. Postprocess data.
python postprocess.py \
	--dataset=toy \
	--num_folds=3 \
	--test_proportion=0.2 \
	--hypothesis_relation=bornIn
