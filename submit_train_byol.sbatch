#!/bin/bash
#SBATCH --job-name=testjob
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=00:05:00
#SBATCH --output=logs/testjob-%j.log
#SBATCH --error=logs/testjob-%j.log

singularity exec /idia/software/containers/ASTRO-GPU-PyTorch.simg python create_embeddings.py
