#!/bin/sh
#SBATCH -N 4
#SBATCH -n 112  
#SBATCH --ntasks-per-node=28 
#SBATCH --partition=normal,normal1,normal2,normal3,normal4  
#SBATCH --output=%j.out  
#SBATCH --error=%j.err   
source /data/app/intel/bin/compilervars.sh intel64   
source ~/3.11/anaconda3/bin/activate
conda activate HEADS
ulimit -s unlimited


# 定义包含Python脚本的目录
script_dir="/data/home/23cj/HEADS/phaseML/HEA_phaseML_train/GridSearchchCV_smote"

# 获取目录下的所有Python脚本
python_scripts=("$script_dir"/*.py)

# 遍历并执行每个Python脚本
for script in "${python_scripts[@]}"; do
    echo "Executing $script..."
    if [ -f "$script" ]; then
        python "$script"
    else
        echo "No Python scripts found in $script_dir"
        exit 1
    fi
done