#!/bin/sh

#SBATCH --time=00-01:00:00 #DD-HH:MM:SS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=15000M
#SBATCH -C icelake
#SBATCH -p gen

#SBATCH --mail-type=all
#SBATCH --mail-user=fm2912@columbia.edu

module load python
source /mnt/home/fmaion/packages/baccogit/.venv/bin/activate 

echo
echo "Running on hosts $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo
echo

srun python /mnt/home/fmaion/storage/projects/bias_priors/scripts/measuring_biases/measure_biases_SB.py 60
