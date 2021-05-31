#!/usr/bin/env bash

output_dir=$1
folds_dirs='fold_0 fold_1 fold_2 fold_3 fold_4'
final_dir='final'
folds_final_dirs="$folds_dirs $final_dir"

# analysis of folds
python3 ../analysis/aggregate_results.py --dir $folds_dirs --results_dir $output_dir
python3 ../analysis/analyze_model_predictions.py --dir $folds_final_dirs  --results_dir $output_dir
