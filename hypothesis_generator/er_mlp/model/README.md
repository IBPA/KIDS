# model

Code to create model instances

## Create a model

```
python3 build_network.py fold_0
```
* a folder to store the model containing .ini file with the same name needs to be created in the model_instance directory. This will take a few hours to run.

## Create thresholds for classification:

```
python3 determine_thresholds.py fold_0
```
* The trained model is ran over the validation set to identify thresholds to use for classification.

## Evaluate the network:
```
python3 evaluate_network.py fold_0
```
* This will evaluate the network over the test dataset. The figures are saved in the model_instance/[[model name]/test directory
