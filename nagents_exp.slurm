#!/bin/bash
#SBATCH --job-name=numagents     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=mlaprise@princeton.edu

module purge
module load cudatoolkit/12.8
nvidia-smi > "./results/nvidia-smi-${SLURM_JOB_ID}.txt"

uv run ./src/nagents_experiment.py --t 50 --m 25 --r 25 --density 0.5 --hidden_dim 128 --lr 1e-4 --epochs 200 --batch_size 200 --steps 5 --train_n 10000 --val_n 400 --test_n 400 --nres 10 --no-sharedv
