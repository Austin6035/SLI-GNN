# SLI-GNN

## Introduction

A Self-Learning-Input Graph Neural Network introduces a dynamic embedding layer to accept the feedback of backpropagation during the training process and introduce the Infomax mechanism to maximize the correlation between the local features and the global features.

## Dependencies

The project is built using the Python language and the following third-party frameworks:

```pyton
PyTorch 1.9.0
PyTorch Geometric 2.0.3
pymatgen 2022.0.17
mendeleev 0.9.0
openbabel 3.1.1
```

## Installation

Before downloading, please ensure that other dependencies have been installed.

First, create a new conda environment

```shell
conda create --name version python=3.7
```

Then, let's install the package from github

```shell
git clone https://github.com/Austin6035/SLI-GNN.git
```

## Example

### Dataset

The material structure dataset should be placed under `data/dataset/` directory, and the target property file should be placed under `data/dataset/prop/`directory, Please note that the material structure file types supported by this project include `cif`, `xyz`, `mol`, `pdb`, `sdf`, and the target property file format should be `csv`.There are already `sample-dataset/` and `sample-targets.csv` in the project as running examples, and other datasets such as QM9 can be downloaded in Kaggle or Material Project.

### Running

Training sample data and other parameter descriptions can be viewed using the following command `python trainer.py -h`. Combined with `ray-tune`, automatic parameter tune can be realized. Depending on the task type, the results will be saved in the `results/regression/` or `results/classification/` directory, and the loss during training will be saved in the `results/` directory, and the log information during training will be saved in the `log/` directory.

```
python trainer.py sample-dataset sample-targets
```

### Testing

After the training is complete, the best model will be saved to the `weight/` directory, and you can use `test.py` for testing. When testing, there can only be a material_id column in the target property file.

```shell
python test.py model_best.pth.tar sample-dataset sample-targets
```

