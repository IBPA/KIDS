#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# directories & files
current_dir=`pwd`
data_dir="$current_dir/data"
output_dir="$current_dir/output"
folds_dir="$output_dir/folds"
final_dir="$output_dir/final"

dr_filename="domain_range.txt"
data_filename="data.txt"
entities_filename="entities.txt"
entity_full_names_filename="entity_full_names.txt"
entity_full_names_copy_filename="entity_full_names_copy.txt"
relations_filename="relations.txt"
hypotheses_filename="hypotheses.txt"

dr_filepath="$data_dir/$dr_filename"
entities_filepath="$output_dir/$entities_filename"
entity_full_names_filepath="$output_dir/$entity_full_names_filename"
entity_full_names_copy_filepath="$output_dir/$entity_full_names_copy_filename"
relations_filepath="$output_dir/$relations_filename"
hypotheses_filepath="$output_dir/$hypotheses_filename"
dr_copy_filepath="$output_dir/$dr_filename"
data_filepath="$output_dir/$data_filename"
data_copy_filepath="$folds_dir/$data_filename"

# variables
num_folds=5

# do post-processing
python3 postprocess_data.py

# process entity_full_names.txt
cp $entity_full_names_filepath $entity_full_names_copy_filepath
sed  -i -E 's|(.+:.+:.+)(:)(.+)|\1#SEMICOLON#\3|g' $entity_full_names_copy_filepath
sed  -i -E 's| |#SPACE#|g' $entity_full_names_copy_filepath
sed  -i -E 's|,|#COMMA#|g' $entity_full_names_copy_filepath

# process entities.txt
sed  -i -E 's|:|#SEMICOLON#|g' $entities_filepath
sed  -i -E 's| |#SPACE#|g' $entities_filepath
sed  -i -E 's|,|#COMMA#|g' $entities_filepath

# process data.txt
cp $data_filepath $data_copy_filepath
sed  -i -E 's|:|#SEMICOLON#|g' $data_copy_filepath
sed  -i -E 's| |#SPACE#|g' $data_copy_filepath
sed  -i -E 's|,|#COMMA#|g' $data_copy_filepath

# process domain_range.txt
cp $dr_filepath $dr_copy_filepath
sed -i '1d' $dr_copy_filepath # remove first line

sed  -i -E 's|:|#SEMICOLON#|g' $dr_copy_filepath
sed  -i -E 's| |#SPACE#|g' $dr_copy_filepath
sed  -i -E 's|,|#COMMA#|g' $dr_copy_filepath

# generate relations.txt file
cut -f 1 $dr_copy_filepath > $relations_filepath

# copy hypotheses to final directory as text.txt file
cp $hypotheses_filepath "$final_dir/test.txt"

# process folds
for ((i=0; i<num_folds; i++)); do
	fold_i="fold_$i"
	copy_to="$folds_dir/$fold_i"

	# process the original files
	find $copy_to -type f -exec sed  -i -E 's|:|#SEMICOLON#|g' {} \;
	find $copy_to -type f -exec sed  -i -E 's| |#SPACE#|g' {} \;
	find $copy_to -type f -exec sed  -i -E 's|,|#COMMA#|g' {} \;

	# copy the common files into each fold directories
	cp $entities_filepath "$copy_to"
	cp $entity_full_names_copy_filepath "$copy_to/$entity_full_names_filename"
	cp $dr_copy_filepath "$copy_to"
	cp $relations_filepath "$copy_to"
done

# process final data
find $final_dir -type f -exec sed  -i -E 's|:|#SEMICOLON#|g' {} \;
find $final_dir -type f -exec sed  -i -E 's| |#SPACE#|g' {} \;
find $final_dir -type f -exec sed  -i -E 's|,|#COMMA#|g' {} \;

# copy the common files into each fold directories
cp $entities_filepath "$final_dir"
cp $entity_full_names_copy_filepath "$final_dir/$entity_full_names_filename"
cp $dr_copy_filepath "$final_dir"
cp $relations_filepath "$final_dir"

# remove temporary files
rm $dr_copy_filepath
rm $entity_full_names_copy_filepath
