#!/bin/bash
set -e

cd ./src

# Step 1. Create the intermediate knowledge graph.
python create_intermediate_kg.py --dataset=ecoli

# Step 2. Check and validate the inconsistencies.

#    Put the validation results into the validation
#    folder (e.g. ./data/ecoli/validation).

# Step 3. Create the inconsistency-free knowledge graph.
python create_inconsistency_free_kg.py --dataset=ecoli

# Step 4. Postprocess data.
python postprocess.py --dataset=ecoli
