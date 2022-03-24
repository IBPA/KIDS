# Knowledge Graph (KG) Constructor
This folder contains source code responsible for **1)** creating the KG using knowledge extracted from the datasets available online and **2)** resolving any inconsistencies that may rise during the KG construction process. The KG created here will be used to train the [Hypothesis Generator](/hypothesis_generator).

## Directories
* <code>[./configuration](./configuration)</code>: Contains **.ini** files used for setting up the configurations.
* <code>[./data](./data)</code>: Contains the dataset used to create the KG. It also contains other data files used as an input for creating the KG. Please refer to its own [README](./data/README.md) file for its own file structure.
* <code>[./integrate_modules](./integrate_modules)</code>: Contains source code for creating the knowledge graph and doing inconsistency resolution.
* <code>[./output](./output)</code>: All output files will end up here.
* <code>[./postprocess_modules](./postprocess_modules)</code>: Contains source code for postprocess.
* <code>[./tools](./tools)</code>: Utility files.

## How to Run
Steps below describe how to replicate the results reported in the paper.

### Step 1: Update the paths.
Running the following script will automatically update the paths to match your local computer path.
```
./update_paths.sh
```

### Step 2: Clean the output directory.
Current <code>[output](./output)</code> directory consists of results used in the paper. If you wish to run the code and obtain new results, please remove all files and directories under it.
```
rm -r ./output/*
```

### Step 3: Run the KG construction and inconsistency resolver.
This step will construct the inconsistency-free knowledge graph. The generated files will be populated under the <code>[output](./output)</code> directory.
```
python3 create_kg.py --phase=all
```

Note that the [default configuration](./configuration/create_kg_config.ini) removes the temporal data from the knowledge graph. In order to create the complete knowledge graph containing the temporal information, please comment out the line as follows:
```
# Comment out this line
replace_rule = /path/to/data/directory/replace_rules.xml

# Like this
# replace_rule = /path/to/data/directory/replace_rules.xml
```

For running a toy example with smaller dataset, update the [configuration](./configuration/create_kg_config.ini) file as below. Running the toy example will take roughly 8 seconds on a normal desktop.
```
# Change this line
data_path = /path/to/data/directorydata_path_file_toy.txt

# Like this
data_path = /path/to/data/directorydata_path_file_toy.txt
```

### Step 4: Run postprocessing.
Postprocess the knowledge graph created in Step 3 to generate hypothesis generator friendly files.
```
./run_postprocess.sh
```

## Output
Following files and directories will be populated under the [output](./output)</code> directory once finished running.

* <code>[final](./output/final)</code>: Folder containing all the files to be used for training the final model.
* <code>[folds](./output/folds)</code>: Folder containing all the files to be used for doing the k-fold cross validation.
* <code>[data.txt](./output/data.txt)</code>: Same file as <code>[kg_final.txt](./output/kg_final.txt)</code> with subset of columns *Subject*, *Predicate*, and *Object*. Also, the *Label* column is added.
* <code>[entities.txt](./output/entities.txt)</code>: All entities of the knowledge graph.
* <code>[entity_full_names.txt](./output/entity_full_names.txt)</code>: All entities of the knowledge graph with their corresponding entity types.
* <code>[hypotheses.txt](./output/hypotheses.txt)</code>: Hypotheses that needs to be generated using the Hypothesis Generator.
* <code>[kg_final.txt](./output/kg_final.txt)</code>: Final knowledge graph produced by the Knowledge Graph Constructor. This file includes the resolved inconsistencies. The Knowledge is represented in the triplet format. There are 6 columns in the following order: ***Subject***, ***Predicate***, ***Object***, ***Belief*** (The confidence score of this specific fact measured by the inconsistency corrector. Default is AverageLog), ***Source size*** (The number of sources supporting this knowledge), and ***Sources*** (The list of sources supporting this knowledge.)
* <code>[kg_without_inconsistencies.txt](./output/kg_without_inconsistencies.txt)</code>: Same file as the [kg_final.txt](./output/kg_final.txt) except for the absence of resolved inconsistencies.
* <code>[relations.txt](./output/relations.txt)</code>: All the relations in the knowledge graph.
* <code>[resolved_inconsistencies.txt](./output/resolved_inconsistencies.txt)</code>: File containing the inconsistencies resolved through the computational method. The first six columns are for the fact with the highest belief (i.e. one that the inconsistency corrector think it is correct) among conflicting facts. The columns are in the following order: ***Subject***, ***Predicate***, ***Object***,***Belief*** (The confidence score of this specific fact measured by the inconsistency corrector. Default is AverageLog.), ***Source size*** (The number of sources supporting this knowledge.), ***Sources*** (The list of sources supporting this knowledge.), ***Total source size*** (The number of all sources including the source size from the conflicting facts.), ***Mean belief of conflicting tuples*** (The average of beliefs of all other tuples conflicting to one that is represented in the first 3 columns.), ***Belief difference*** (The difference between the *Belief* and the *Mean belief of conflicting tuples*.), and ***Conflicting tuple info*** (The list of all the tuples conflicting to one that is represented in the first 3 columns. It is the list represented in "[(element1), (element2), ...]" and each element of the list is represented by "tuple, sources, belief" where tuple is "(subject, predicate, object)", and source is "[source1, source2, ...]".)
* <code>[trustworthiness_data_summary.pdf](./output/trustworthiness_data_summary.pdf)</code>: Figure showing the statistics of the knowledge base integration result.
* <code>[validated_inconsistencies.txt](./output/validated_inconsistencies.txt)</code>: Inconsistencies that has been resolved using the computational method and wet-lab validation.
