#!/bin/bash
set -e

cd ./src/TuckER

# hyperparameter search
python kids.py \
	--dataset=ecoli \
	--mode=evaluate \
	--random_state=530 \
	--num_iterations=300 \
	--batch_size=128 \
	--lr=0.0002 \
	--dr=1.0 \
	--edim=200 \
	--rdim=100,30 \
	--cuda=True \
	--input_dropout=0.2 \
	--hidden_dropout1=0.4 \
	--hidden_dropout2=0.5 \
	--label_smoothing=0.1 \
	--output_filename=hyperparameter_search_results.pkl

# find best hyperparameter
python analyze_results.py \
	--mode=find_best_hyperparameter \
	--input_filename=hyperparameter_search_results*.pkl

# use best hyperparameter and run on the test set
python kids.py \
	--dataset=ecoli \
	--mode=evaluate \
	--random_state=530 \
	--num_iterations=300 \
	--batch_size=128 \
	--lr=0.0002 \
	--dr=1.0 \
	--edim=200 \
	--rdim=30 \
	--cuda=True \
	--input_dropout=0.2 \
	--hidden_dropout1=0.4 \
	--hidden_dropout2=0.5 \
	--label_smoothing=0.1

# integrate test results
python analyze_results.py \
	--mode=get_test_stats \
	--input_filename=evaluation_results.pkl

# use best hyperparameter and generate hypothesis
python kids.py \
	--dataset=ecoli \
	--mode=final \
	--random_state=530 \
	--num_iterations=5 \
	--batch_size=128 \
	--lr=0.0002 \
	--dr=1.0 \
	--edim=200 \
	--rdim=30 \
	--cuda=True \
	--input_dropout=0.2 \
	--hidden_dropout1=0.4 \
	--hidden_dropout2=0.5 \
	--label_smoothing=0.1
