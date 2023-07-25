## Title: 
EfficientNetV2 Neural Architecture Search - Project Artefacts
November 2022

## Author:
Adam Peaston
SID: XXX XXX XXX
Unikey: xxxxXXXX

## Description:
This package contains files created in the course and for the purpose of the Data Science Capstone Project "DATA5709" titled "EfficientNetV2 Neural Architecture Search".
The contents of this package is provided as Appendix to the final report of the same title which provides further explanation and discussion of the techniques demonstrated.

## Getting started:
This project was developed using Docker for containerisation. It is recommended to open this direcory in a container to ensure environment is properly initialized.

## Contents:
final_package
|-- README.txt
|-- Dockerfile
|-- components.py
|-- bte_ddp.py
|-- framerate_ddp_INCOMPLETE.py
|-- BetaNet_NAS.ipynb
|-- BetaNet_NAS_results_analysis.ipynb
|-- BetaNet2_evaluation.ipynb
|-- BetaNet3_evaluation.ipynb
|-- benchmark_training.ipynb
|-- plotting_benchmarks.ipynb
|-- nas_trial_results
|	|-- beta2
|	|	|-- R224
|	|		|-- trial_results
|	|-- beta3
|		|-- R224
|			|-- trial_results
|-- experiment_results
|	|-- ...
|-- nas_trial_resukts
	|-- NULL.txt

## Usage overview
Essential components of the BetaNet2 and BetaNet3 models such as custom layers and models, including benchmark model constructors are found in the 'components.py' file.
The machinery of distributed training is found in the 'bte_ddp.py' script (acronym stands for "build, train, evaluate, distributed data parallel".
Each of the jupyter notebooks for 'NAS', 'training', and 'evaluation' call the bte_ddp.py script via the notebook magic command "%run -i bte_ddp.py" where the "-i" argument
specifies that the script should inherit the namespace from the jupyter notebook, including all necessary input arguments (refer to bte_ddp.py for a list of these arguments).
The bte_ddp.py script includes a '__main__' process which calls the training function.

## Reproducing the NAS results
The user can open the "BetaNet_NAS.ipynb" notebook and run this end-to-end to reproduce the Neural Architecture Search of BetaNet2 or BetaNet3 by providing the appropriate arguments.
Results of the NAS are deposited in the "nas_trial_resukts" and should be transferred to a suitable sub-directory within nas_trial_results before commencing another NAS routine.
Results of NAS can then be analysed and plotted using the "BetaNet_NAS_results_analysis.ipynb" notebook.
BetaNet models and benchmark models can be trained for evaluation using the "BetaNet{i}_evaluation.ipynb" and "benchmark_training.ipynb" notebooks respectively.
Results of evaluation, including saved model parameters and performance results, are saved in the "experiment_results" directory.
Benchmark comparisons with BetaNet2/3 models are prepared using the "plotting_benchmarks.ipynb" notebook.
A supplementary file is included "framerate_ddp_INCOMPLETE.py" which was partially complete at the time of submission. The intention of this script was to test and compare inference
rates for the BetaNet2/3 models with those for the benchmark models. The user is invited to continue this work if it is of interest.

