#!/usr/bin/env bash

start_relation="gender"
DATA_PATH="/Users/nicholasjoodi/Documents/ucdavis/research/HypothesisGeneration/data/freebase/"
use_negatives=false
use_domain=false
no_negatives=('has' 'is' 'is#SPACE#involved#SPACE#in' 'upregulated#SPACE#by#SPACE#antibiotic' 'targeted#SPACE#by' 'activates' 'represses'  )
train_file="train.txt"
log_reg_calibrate=true
use_smolt_sampling=true
predict_file=train_local.txt

