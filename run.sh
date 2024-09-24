#!/bin/bash
#SBATCH --job-name=Swinv2b_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4         # Request X CPUs for your task
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=16384M
#SBATCH --time=08-00:00:00
#SBATCH --account=def-velab
#SBATCH --mail-type= END,FAIL
#SBATCH --mail-user=ahmed.dhahri@uqtr.ca
#SBATCH --output=output_s.txt
#SBATCH --error=error_s.txt

module load StdEnv/2020
module load gcc/9.3.0
module load cuda/11.4
module load python/3.10
module load opencv/4.7.0

python3 Material_recognition/train/train_multi_thread.py swinv2b False

 

# how to :


# sbatch run.sh
# This is a comment in a shell script
# Check the status of all jobs for a user: squeue -u karem
# Check the status of a specific job: squeue -j 6329147
# To remove a job from the SLURM queue: scancel 6329147
