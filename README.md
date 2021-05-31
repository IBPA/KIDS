# Knowledge Integration and Decision Support (KIDS)
KIDS constructs an inconsistency-free knowledge graph that supports multiple triple types and performs knowledge graph completion. We apply the KIDS framework to the area of *Eschericia coli* antibiotic resistance. This work proposes an integrated approach towards automated knowledge representation and discovery, and demonstrates how evidence-driven decisions can automate knowledge discovery with high confidence and accelerated pace.

![Figure 1](/images/Figure1.png)
*Figure 1. Overview of the KIDS framework.*

## Directories
* <code>[./hypothesis_generator](./hypothesis_generator)</code>: Code for generating the hypothesis based on the knowledge graph created. Please refer to its own [README](/hypothesis_generator/README.md) file for more information.
* <code>[./images](./images)</code>: Contains README file related images.
* <code>[./kg_constructor](./kg_constructor)</code>: Code for creating the knowledge graph. Note that this directory also contains the source code for performing inconsistency resolution. Please refer to its own [README](./kg_constructor/README.md) file for more information.
* <code>[./manuscript_preparation](./manuscript_preparation)</code>: Code for analyzing the results as reported in the manuscript.

## Getting Started

This code has been tested with Python 3.6 under both Ubuntu 18.04 LTS and Ubuntu 20.04 LTS.

### 2a. Clone this repository to your local machine.

```
mkdir KIDS
git clone https://github.com/IBPA/KIDS.git ./KIDS
```

### 2b. Install all the dependencies.

Create and activate virtual environment.
```
cd ./KIDS
python3 -m venv env
source env/bin/activate
```

Install all required python packages once the virtual environment has been activated.
```
pip3 install -r requirements.txt
```

You will also need Java 7 or higher. If you are running Ubuntu 18.04, follow the steps below to install Java OpenJDK 11.
```
sudo apt update
sudo apt install openjdk-11-jdk
```

(Optional) You can deactivate the virtual environment once finished.
```
cd ./KIDS
deactivate
```

### 2c. Running the code.
- Construct the KG by following the [README](/kg_constructor/README.md) file.
- Generate the hypothesis by following the [README](/hypothesis_generator/README.md) file.

## 3. Contact

For any questions, please contact us at tagkopouloslab@ucdavis.edu.

## 4. Citation

We will update this section once citation information is available.

## 5. License

This project is licensed under the **Apache-2.0 License**. Please see the <code>[LICENSE](./LICENSE)</code> file for details.

## 6. Acknowledgments

* Special thanks to the members of the [Tagkopoulos lab](http://tagkopouloslab.ucdavis.edu/) and the reviewers for their suggestions.
* Nick Joodi and Minseung Kim for their help in the initial discussions, and Ameen Eetemadi for his comments on creating the figures.
* This work was supported by the USDA-NIFA AI Institute for Next Generation Food Systems (AIFS), USDA-NIFA award number 2020-67021-32855 and the NIEHS grant P42ES004699 to Ilias Tagkopoulos.
