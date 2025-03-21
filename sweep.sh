#!/bin/bash
#SBATCH -o /home/mila/m/marco.stock/slurm_logs/slurm-%j.out
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=128G                     # server memory requested (per node)
#SBATCH -t 3:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=short-unkillable               # Request partition
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100l:1                 # Type/number of GPUs needed
#SBATCH -c 6
#SBATCH --open-mode=append            # Do not overwrite logs

# option: SBATCH --requeue                     # Requeue upon pre-emption

# 1. Load the required modules
module --quiet load miniconda/3

# 2. Load your environment
conda activate ptgeo

# 3. Launch your job
cd /home/mila/m/marco.stock/granpy/
srun wandb agent scialdonelab/granpy-dev/vtr3u2la