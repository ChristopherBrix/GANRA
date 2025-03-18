# GANRA

This is the code for the VerifAI workshop paper `Using GPUs And LLMs Can Be Satisfying for Nonlinear Real Arithmetic Problems`.

## Cloning

To clone the repository, make sure to set the `--recurse-submodules` flag. Otherwise, after cloning, run `git submodule update --init --recursive`.

## Installation

### Dependencies
Install the venv environment using
```bash 
python3.12 -m venv venv 
source venv/bin/activate
# install all needed requirements, this step might take a few minutes
pip install -r requirements.txt
```

### CVC5
If you want to run the verification of generated SAT instances using CVC5, follow the [Installation guide](https://github.com/cvc5/cvc5/blob/main/INSTALL.rst) and place it in the parent directory of this repository.

### ugotNL
The source code of ugotNL is not currently public. We will update this installation instruction as soon as it is.
Feel free to open an issue in this repository so we can notify you as soon as that's done.


## Replication of experiments

To run the experiments, you need to uncomment the desired options in array_jobs/experiments.py 

Depending on the benchmarks you want to run experiments on, you need to create some of the following files: `random_files_mbo_orig.txt`, `random_files_mbo_preprocessed.txt`, `random_files_kissing.txt`. Each of them contains the paths to the files considered for the experiments. One line should correspond to one path. 

Next, apply the following command to create a task list which later will be worked through by multiple jobs. 

```bash 
python array_jobs/experiments.py <optional_flags> > tasks.txt
```
The optional_flags can be either `--generate_custom_mbo` or `--evaluate_custom_mbo` as specified in `array_jobs/experiment.py`. 
The number of jobs running can be modified in `array_jobs/run_all.sh`. 

Then set up array jobs to execute the experiments as fast as possible. 
```bash 
sbatch --gpus-per-node=1 --cpus-per-task=16 array_jobs/run_all.sh
```
Note: This assumes you have SLURM installed. Alternatively, you can run `array_jobs/run_all.sh` locally, too.

## Running new experiments
Extending this framework to another benchmark is easily possible by first creating a subclass of `GeneralAttackModel` which implements initialization and forward passes. Next, the `attacker.py` needs to be slightly updated to handle another forward logic. 
Moreover, to generate GPT based implementations, the following command can be used: 
```bash 
python3 gpt_input_generation.py <file.smt2> > gpt_input.txt 
```
The script translates the benchmark into a more compact and machine-readable layout and formulates the prompt which can be used as the input to the LLM. 
