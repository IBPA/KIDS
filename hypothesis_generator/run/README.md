# run

This directory contains shell scripts to create, train, and evalue models. Note: Each experiment has its configurations set in the configurations folder. To observe a description of each parameter that you can set for each model, oberve the configurations for the fold_0 experiment. Comments are provided.

## run_er_mlp.sh
To create, train, and evaluate the ER-MLP over a particular experiment, run:

```
./run_er_mlp.sh dir_name
```
where dir_name represents the folder containing the er_mlp configurations. This folder is stored in the configurations subdirectory. For example, to run er_mlp over the first fold of the experiments for this project:
```
./run_er_mlp.sh fold_0
```
Notice the fold_0 directory in the configurations directory.The configuration file in this case is the er_mlp.ini file. When this script runs, the model will be created, trained, and evaluated over the fold 0 data set for this project. The model and its associated files will be automatically created in [root]/er_mlp/model/model_instance/fold_0. Note: before running this script, the DATA_PATH property in the er_mlp.ini file will need to be updated to where the dataset is stored on your machine.

## run_pra.sh
To create, train, and evaluate the PRA over a particular experiment, run:

```
./run_pra.sh dir_name
```
Similarly to the er_mlp, the dir_name represents the folder containing the PRA configurations. In this case, the configuration for the PRA is separated into two files. The conf file is the configuration file that is provided by the author of the original code of the PRA. This conf file contains hyper parameter configurations. The config.sh file contains configurations that we added for our experiments. For example, to run PRA over the first fold of the experiments for this project:
```
./run_pra.sh fold_0
```
The model and its associated files will be automatically created in [root]/pra/model/model_instance/instance/fold_0. Note: before running this script, the DATA_PATH property in the config.sh file will need to be updated to where the dataset is stored on your machine.

## run_stacked.sh
After a pra and er_mlp have been created, train, and evaluated, a stacked ensemble of their scores can be combined to. We can create, train and evaluate a stacked ensemble by running the following command.

```
./run_stacked.sh dir_name
```
This works identically to the PRA and the ER-MLP with the exception that the configuration file is now called stacked.ini. An example of running this command:
```
./run_pra.sh fold_0
```
The model and its associated files will be automatically created in [root]/stacked/model_instance/fold_0. Note: This script can only be ran after the PRA and er-mlp scripts were ran above.

## run_report.sh

After the above experiments have been run, you can then execute the run_report script. Note: this feature has been implemented for the project experiments since we are performing k fold cross validation. For results of freebase, those results can be found in the test directory of the actual model instance. A results.txt along with figures will are provided for each model instance.

```
./run_report.sh --results_dir [[results folder]] --dir [[space separated experiment folders]]
```
This will create a new directory using the name provided by the results_dir argument providing evaluation metrics over the experiments provided after the --dir argument. For example:
```
 ./run_report.sh --results_dir results --dir fold_0 fold_1 fold_2
```
will create ROC, PR curves, confusion matrix, and other interesting metrics in the results folder created.


