#!/bin/bash
#SBATCH --job-name=testjob
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/testjob-%j.out
#SBATCH --error=logs/testjob-%j.err

## IF YOU WANT TO CHANGE WEIGHTING, PROB, ADD THESE OPTIONS TO THE PYTHON CALL:
## --weighting=my_choice --prob=my_value
singularity exec /idia/software/containers/ASTRO-GPU-PyTorch.simg python create_embeddings.py --run-name=$1

## TO SUBMIT A JOB DO:
## sbatch submit_train_byol.sbatch <this_run_name>

