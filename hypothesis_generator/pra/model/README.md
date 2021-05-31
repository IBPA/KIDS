# model

Code to create model instances

## Create a model

```
./build_models.sh  fold_0/
```
* a folder to store the model with the same name needs to be created in the model_instance directory. Additionally, there are configuration files that need to be created before creating the model.

## Create thresholds for classification:

```
./determine_thresholds.sh fold_0/
```
* The trained model is ran over the validation set to identify thresholds to use for classification.

## Evaluate the network:
```
evaluate_models.sh fold_0/
```
* This will evaluate the network over the test dataset. The figures are saved in the model_instance/[model name]/instance/test directory
