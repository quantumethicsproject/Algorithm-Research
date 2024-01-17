#### How to navigate this directory
I've broken up the code into a main notebook (`VQLS_Jeffrey.ipynb`) and various modules for the various parts of the VQA. Here is a high-level overview of the structure:

#### `bin`
This directory contains all the utilities.
* `hyperparameters.py`
  * This controls the hyperparameters of the experiment. Nearly all of my modules import this file.
* `cost_function.py`
  * Contains the code for the cost function. `cost_local` and `cost_global` are the functions that the main notebook uses, and the others are the code implementations of the various terms defined in the paper.
* `error_mitigation.py`
  * Contains wrapper functions for mitiq error correction
* `inference.py`
  * Contains helper functions to interpret results (it's more like a sanity check)

#### `problems`
This is the folder that contains the problem class. `problem_base.py` is an abstract base class that defines the structure and interface that all problem classes should implement. `ising_problem2.py` implements the Ising problem and `toy_problem.py` implements simple toy problems described in the paper. `vqls.py` is a file with useful helper functions that parses matrix representations into numpy arrays etc.



Some instructions for running the notebook:

1. Create a virtual environment using the following command in PowerShell
```
py -m venv directory_name
```
1. Initialise the virtual environment using the following command in PowerShell
```
directory_name\Scripts\Activate.ps1
```
1. Run the following command with your virtual environment active to install all required dependencies
```
py -m pip install -r requirements.txt
```
1. Check that the notebook's Python kernel is switched to the virtual environment Python kernel!


For dev work:

- Remember to run the following command in your virtual environment (it'll save installed packages used in dev work)
```
py -m pip freeze > requirements.txt
```