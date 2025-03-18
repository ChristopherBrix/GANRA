#!/bin/bash

#SBATCH --array=0-3
#SBATCH --ntasks=1 # n tasks, may be split across multiple CPUs
#SBATCH --time=0-01:00:00

source venv/bin/activate

python3 array_jobs/run_all.py 

