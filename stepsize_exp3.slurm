#!/bin/bash
#SBATCH --job-name=stepsize1e5   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=mlaprise@princeton.edu

module purge
module load cudatoolkit/12.8
nvidia-smi > "nvidia-smi-${SLURM_JOB_ID}.txt"

uv run ./src/stepsize_experiment.py --t 50 --m 25 --r 25 --density 0.5 --num_agents 128 --hidden_dim 384  --lr 1e-5 --epochs 150 --batch_size 200 --train_n 4000 --val_n 300 --test_n 300 --nres 10