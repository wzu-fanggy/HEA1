#!/bin/sh
#SBATCH -N 4
#SBATCH -n 112
#SBATCH --ntasks-per-node=28
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --output=%j.out
#SBATCH --error=%j.err
source /data/app/intel/bin/compilervars.sh intel64
mpirun -np $SLURM_NPROCS /data/app/lammps-8Feb2023/src/lmp_intel_cpu_intelmpi < strain_stress.in
#（输入文件请用户根据实际情况自行修改）