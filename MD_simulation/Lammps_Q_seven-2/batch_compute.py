import os
import shutil
import time
import re
import numpy as np



pattern = r"([A-Z][a-z]*)(0\.\d+)"

def set_pair_coeff_eam(alloy):
    '''
    alloy = 'Fe0.20Ni0.20Cr0.20Co0.20Mg0.20_fcc.lammps-data'
    '''
    matches = re.findall(pattern, alloy)
    elements, fractions = zip(*matches)
    fractions = np.array([float(f) for f in fractions])
    # 将元素元组转换为字符串，并用空格连接
    elements_str_file = ''.join(elements)
    elements_str = ' '.join(elements)
    # 创建最终的字符串格式
    pair_coeff_eam_name = f"{elements_str_file}.eam.alloy"
    final_str_in = f"pair_coeff * * {elements_str_file}.eam.alloy {elements_str}\n"
    return pair_coeff_eam_name, final_str_in




comfile_path = 'Comfile'
comfile_absolute_path = os.path.abspath(comfile_path)
source_files = ['lammps_strain_stress.sh', 'strain_stress.in']  
source_files_am = ['lammps_strain_stress_am.sh', 'strain_stress_am.in']  

hea_eam_files = 'HEA_eam'
hea_eam_absolute_path = os.path.abspath(hea_eam_files)

final_struct = 'Final_structures'

final_struct_file_absolute_path = os.path.abspath(final_struct)
# 获取当前目录下所有以 "_final" 结尾的 data 文件
data_files = [os.path.join(final_struct_file_absolute_path, f) for f in os.listdir(final_struct_file_absolute_path) if f.endswith('data')]
submit_count=0

# 遍历所有文件
for data_file in data_files:
    print(data_file)
    # data_file是绝对路径，Final_struct文件夹下所对应的.data文件
    # 提取文件名（不包括路径）
    file_name = os.path.basename(data_file)
    
    print('file_name: ',file_name)
    # 移除 ".data" 后缀以获取文件夹名称
    folder_name = file_name[:-12]  # 移除 ".data" 后缀
    print(folder_name)
    # 获取文件夹的绝对路径
    subfolder_absolute_path = os.path.join(final_struct_file_absolute_path, folder_name)
    print('subfolder_absolute_path', subfolder_absolute_path)
    # 如果文件夹不存在，则创建对应的文件夹
    if not os.path.exists(subfolder_absolute_path):
        os.makedirs(subfolder_absolute_path)
    
    #将对应的势函数文件存入对应的文件夹
    pair_coeff_eam_name, final_str_in = set_pair_coeff_eam(folder_name)
    shutil.copy(os.path.join(hea_eam_absolute_path, pair_coeff_eam_name), subfolder_absolute_path)
    
    # 复制 data 文件到新建文件夹
    shutil.copy(data_file, subfolder_absolute_path)
    

    if folder_name.endswith('am'):
        # 复制 Comfile 文件夹中的文件到新建文件夹,非晶合金需要退火
        for file in source_files_am:
            shutil.copy(os.path.join(comfile_absolute_path, file), subfolder_absolute_path)
        # 修改 strain_stress_am.in 文件
        strain_stress_path = os.path.join(subfolder_absolute_path, 'strain_stress_am.in')
        
        with open(strain_stress_path, 'r') as file:
            lines = file.readlines()
    else:
        # 无需退火时，复制 Comfile 文件夹中的文件到新建文件夹
        for file in source_files:
            shutil.copy(os.path.join(comfile_absolute_path, file), subfolder_absolute_path)
        # 修改 strain_stress.in 文件
        strain_stress_path = os.path.join(subfolder_absolute_path, 'strain_stress.in')
        with open(strain_stress_path, 'r') as file:
            lines = file.readlines()

       
    print('strain_stress_path:', strain_stress_path)
    with open(strain_stress_path, 'w') as file:
        for line in lines:
            if 'variable sub_folder_name'in line:
                line = f'variable sub_folder_name string "{folder_name}"\n'
                print('line',line)
            elif 'variable data_file_name' in line:
                line = f'variable data_file_name string "{folder_name}.lammps-data"\n'
                print('line',line)
            elif 'pair_coeff' in line:
                line = final_str_in
                print('line',line)
            file.write(line)

    # 切换到对应文件夹
    os.chdir(subfolder_absolute_path)
    # 使用 sbatch 提交 lammps_strain_stress.sh 脚本
    
    if folder_name.endswith('am'):
        os.system('sbatch lammps_strain_stress_am.sh')
    else:
        os.system('sbatch lammps_strain_stress.sh')

    submit_count += 1  # Increment the counter
            
    # Pause after every 100 submissions for 15 minutes
    if submit_count % 200 == 0:
        print("Submitted 100 scripts. Pausing for 15 minutes to avoid cluster overload.")
        time.sleep(1800)  # 15 minutes = 900 seconds