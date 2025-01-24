import numpy as np
import pandas as pd
from ase.build import bulk
from ase.io import write
from ase import Atoms
from ase.visualize import view
import numpy as np
from scipy.spatial import Voronoi, cKDTree
import os
import re


def save_structure(alloy, structure, folder_path, structure_type, index, file_format):
    filename = os.path.join(folder_path, f"{alloy}_{structure_type}_structure_{index+1}.{file_format}")
    write(filename, structure, format=file_format)  # 保存为指定格式
    

def save_structure1(alloy, structure, folder_path, structure_type, index, file_format):
    filename = os.path.join(folder_path, f"{alloy}_{structure_type}_structure_{index+1}.{file_format}")
    print('filename',filename)
    write(filename, structure)
    #write(filename, structure, format=file_format)  # 保存为指定格式
    
def save_structure_for_one(alloy, structure, folder_path, structure_type,  file_format):
    filename = os.path.join(folder_path, f"{alloy}_{structure_type}.{file_format}")
    write(filename, structure, format=file_format)  # 保存为指定格式
    
# 生成固定位置的种子点
def generate_fixed_points(num_points, space_dimensions):
    return np.random.rand(num_points, 3) * space_dimensions

# 生成Voronoi图
def voronoi_3d(points):
    return Voronoi(points)

# 为每个整数点分配Voronoi区域和类别
def assign_voronoi_regions_and_categories(vor, space_dimensions):
    x_range = np.arange(0, space_dimensions[0], 1)
    y_range = np.arange(0, space_dimensions[1], 1)
    z_range = np.arange(0, space_dimensions[2], 1)

    grid_points = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)

    # 使用KD树查找最近的Voronoi点
    tree = cKDTree(vor.points)
    distances, indices = tree.query(grid_points)

    categories = np.random.choice(['fcc', 'bcc'], size=grid_points.shape[0])
    return grid_points, indices, categories


def repeatXYZ(target_size_angstrom, init_element_lattice):
    '''
    target_size: 指定模型的长宽高
    '''
    num_repeats_x = np.floor(target_size_angstrom[0] / init_element_lattice).astype(int)
    num_repeats_y = np.floor(target_size_angstrom[1] / init_element_lattice).astype(int)
    num_repeats_z = np.floor(target_size_angstrom[2] / init_element_lattice).astype(int)
    return  num_repeats_x, num_repeats_y, num_repeats_z



def get_init_structure_param(elements, phase, target_size=np.array([1.0, 1.5, 2.0])):
    '''
    获得初始的晶胞参数：初始元素、相结构、晶格常数
    # phase：获得csv文件中，预测得到的相结构
    # target_size目标尺寸，单位为 nm
    '''
    # 首先进行相的映射# 初始相结构
    phase_dict = {0: 'am', 1: 'bcc', 2: 'fcc', 3: 'fccbcc'}
    structure_type = phase_dict[phase]
    
    # 如果是 AM 或 fccbcc，则默认使用 fcc 的晶格参数
    temp_structure_type = structure_type
    if structure_type in ['am', 'fccbcc']:
        temp_structure_type = 'fcc'
    
    # 选择最大晶格常数的元素
    init_element = max(elements, key=lambda el: lattice_constants[el][temp_structure_type])
    
    # 初始晶格常数
    init_element_lattice = lattice_constants[init_element][temp_structure_type]

    # print(f"参数：{init_element},{init_element_lattice}")
   
    target_size_angstrom = target_size * 10  # 转换为 Å

    return init_element, structure_type, init_element_lattice, target_size_angstrom




def check_equal_ratios(element_string):

    # 使用正则表达式提取元素及其比例
    pattern = r'([A-Za-z]+)(\d+\.\d+)'
    matches = re.findall(pattern, element_string)
    
    # 提取比例并转换为浮点数
    ratios = [float(ratio) for _, ratio in matches]
    
    # 检查比例是否相等
    return all(ratio == ratios[0] for ratio in ratios)

    
    
def build_target_size_model(init_element, structure_type, init_element_lattice, target_size_angstrom):
    '''
    构建一个指定初始元素、指定长宽高的模型,已经进行缩放。注意没有进行元素的随机替换
    '''
    print('structure_type:', structure_type)
    print('init_element_lattice: ', init_element_lattice)
    if structure_type == 'fcc':
        # 计算初始晶胞的重复次数
        num_repeats_x, num_repeats_y, num_repeats_z = repeatXYZ(target_size_angstrom, init_element_lattice)
        initial_structure = bulk(init_element, structure_type, init_element_lattice, cubic=True).repeat((num_repeats_x, num_repeats_y, num_repeats_z))
        print_model_info(initial_structure)
    elif structure_type == 'bcc':
        if init_element_lattice <= 0:
            init_element_lattice = lattice_constants['Cr']['bcc']
        num_repeats_x, num_repeats_y, num_repeats_z = repeatXYZ(target_size_angstrom, init_element_lattice)
        initial_structure = bulk(init_element, structure_type, init_element_lattice, cubic=True).repeat((num_repeats_x, num_repeats_y, num_repeats_z))
        print_model_info(initial_structure)
    elif structure_type == 'am':
        max_displacement = 0.02 * init_element_lattice
        structure_type = 'fcc'
        num_repeats_x, num_repeats_y, num_repeats_z = repeatXYZ(target_size_angstrom, init_element_lattice)
        initial_structure = bulk(init_element, structure_type, init_element_lattice, cubic=True).repeat((num_repeats_x, num_repeats_y, num_repeats_z))
        initial_structure = disturb(initial_structure, max_displacement)
        print_model_info(initial_structure)
    elif structure_type == 'fccbcc':
        # 创建FCC单元
        fcc_unit = bulk(init_element, 'fcc', lattice_constants[init_element]['fcc'], cubic=True)
        #fcc_unit_lattice_constants = lattice_constants[init_element]['fcc']
        # 检查BCC参数
        if lattice_constants[init_element]['bcc'] > 0:
            bcc_unit = bulk(init_element, 'bcc', lattice_constants[init_element]['bcc'], cubic=True)
            bcc_unit_lattice_constants = lattice_constants[init_element]['bcc']
            scaling_factor = lattice_constants[init_element]['bcc'] / lattice_constants[init_element]['fcc']
        else:
            # 使用Cr的晶格常数作为默认值
            bcc_unit = bulk('Cr', 'bcc', lattice_constants['Cr']['bcc'], cubic=True)
            bcc_unit_lattice_constants = lattice_constants['Cr']['bcc']
            scaling_factor = lattice_constants['Cr']['bcc'] / lattice_constants[init_element]['fcc']
        
        fcc_unit.set_cell(fcc_unit.cell * scaling_factor, scale_atoms=True)
        print('bcc_unit_lattice_constants:', bcc_unit_lattice_constants)
        # 计算初始晶胞的重复次数
        num_repeats_x, num_repeats_y, num_repeats_z = repeatXYZ(target_size_angstrom, bcc_unit_lattice_constants)
        space_dimensions = [num_repeats_x, num_repeats_y, num_repeats_z]
        num_points = 10  # 种子点数量
        points = generate_fixed_points(num_points, space_dimensions)  # 生成固定位置的种子点
        vor = voronoi_3d(points)  # 生成Voronoi图
        grid_points, indices, categories = assign_voronoi_regions_and_categories(vor, space_dimensions)  # 分配Voronoi区域和类别
        
        #print('space_dimensions:', space_dimensions)
        #print('bcc_unit_lattice_constants', bcc_unit_lattice_constants)
        
        categories = categories.reshape((num_repeats_x, num_repeats_y, num_repeats_z))  # 设置categories为三维数组

        # 创建空模型
        #combined_structure = np.empty((num_repeats_x, num_repeats_y, num_repeats_z), dtype=object)

        # 创建一个空的Atoms对象
        initial_structure = Atoms()
        print('456')
        print(f"fcc_unit模型大小: {fcc_unit.get_cell()} Å")   
        print(f"bcc_unit模型大小: {bcc_unit.get_cell()} Å")   
        # 随机填充FCC和BCC晶胞并合并
        for i in range(num_repeats_x):
            for j in range(num_repeats_y):
                for k in range(num_repeats_z):
                    if categories[i, j, k] == 'fcc':
                        atoms = fcc_unit.copy()
                    else:
                        atoms = bcc_unit.copy()  # 始终使用bcc_unit
                    translation = [i * bcc_unit_lattice_constants, j * bcc_unit_lattice_constants, k * bcc_unit_lattice_constants]
                    #print(f"Translation: {translation}")

                    atoms.translate(translation)
                    #print(f"Before extending, atoms positions: {atoms.get_positions()}")

                    initial_structure.extend(atoms)
                    #print(f"Current structure size after adding atoms: {len(initial_structure)}")
            #print(f"模型大小: {initial_structure.get_cell()} Å")
            
        initial_cell_size = np.array([bcc_unit_lattice_constants] * 3)  # 假设XYZ方向对应相同的晶格常数
        initial_structure.set_cell(initial_cell_size, scale_atoms=False)
        positions = initial_structure.get_positions()
        new_positions = positions / num_repeats_x
        initial_structure.set_positions(new_positions)
        print(f"模型大小: {initial_structure.get_cell()} Å")
    # 最终缩放(根据对角线进行缩放)
    current_size = initial_structure.get_cell().diagonal()  # 获取对角线长度
    scale_factor = target_size_angstrom / current_size  # 计算缩放因子
    initial_structure.set_cell(initial_structure.get_cell() * np.diag(scale_factor), scale_atoms=True)
    print(f'x:{num_repeats_x}, y:{num_repeats_y},z:{num_repeats_z}')
    print(f"模型大小: {initial_structure.get_cell()} Å")    
    return initial_structure


def build_target_size_random_model(initial_structure, elements, fractions):
    '''
    进行元素的随机替换，按照比例
    输入参数：模型，合金的比例
    输出参数：构建好的模型，元素已经进行随机替换
    '''
    # 创建元素列表
    num_atoms = len(initial_structure)
    element_counts = np.array([int(round(num_atoms * fraction)) for fraction in fractions], dtype=int)

    # 校正由四舍五入导致的原子总数不匹配
    if element_counts.sum() != num_atoms:
        element_counts[-1] += num_atoms - element_counts.sum()

    # 填充结构
    shuffled_elements = np.random.choice(np.repeat(elements, element_counts), num_atoms, replace=False)
    initial_structure.set_chemical_symbols(shuffled_elements)

    # 输出信息
    #unique_elements, counts = np.unique(shuffled_elements, return_counts=True)
    #element_count_dict = dict(zip(unique_elements, counts))
    #print(f"合金: {alloy}, 原子总数: {num_atoms}, 各类原子数量: {element_count_dict}, 模型大小: {initial_structure.get_cell()} Å")
    return initial_structure




def disturb(initial_structure, max_displacement):
    '''
    功能：非晶相模型，进行原子位置的扰动
    输入参数：初始模型，最大扰动长度
    '''
    for atom in initial_structure:
        displacement = np.random.uniform(-max_displacement, max_displacement, size=3)  # 随机扰动
        #print('displacement:', displacement)
        #print('atom.position_first', atom.position)

        atom.position += displacement  # 更新原子位置
        #print('atom.position_scaled:', atom.position)
    return initial_structure



def print_model_info(structure):
    print('print_model_info_size:', structure.get_cell())
    print('atoms number:', len(structure))





pattern = r"([A-Z][a-z]*)(0\.\d+)"


# 参数设定
lattice_constants = {
    'Fe': {'fcc': 3.54, 'bcc': 2.87},
    'Ni': {'fcc': 3.52, 'bcc': 0},
    'Cr': {'fcc': 3.58, 'bcc': 2.88},
    'Mn': {'fcc': 3.61, 'bcc': 2.90},
    'Cu': {'fcc': 3.615, 'bcc': 0},
    'Co': {'fcc': 3.54, 'bcc': 0},
    'Mg': {'fcc': 0, 'bcc': 0} ,
    'Ti': {'fcc': 0, 'bcc': 0} ,
    'Al': {'fcc': 4.05, 'bcc': 0}  # Al 通常为 FCC，没有 BCC 结构
}


input_csv = 'eight_HEA_all_proportion_Data.csv'  # CSV文件路径

# 读取CSV文件
df = pd.read_csv(input_csv)

phase_counts = df['Phases considered'].value_counts()
#print('phase_counts:', phase_counts)


main_folder = '/data/home/23cj/HEADS/phaseML/HEA_model_build/HEA20241020_final'
os.makedirs(main_folder, exist_ok=True)


target_size = np.array([10.0, 10.0, 10.0])

'''
main_folder：确定模型需要存放的路径

target_size：确定模型的大小
'''

for index, row in df.iterrows():
    #if index>20:
    #    break
    alloy = row['Alloy']
    #if not check_equal_ratios(alloy):
    #    continue

    matches = re.findall(pattern, alloy)
    
    
    elements, fractions = zip(*matches)
    fractions = np.array([float(f) for f in fractions])
    phase = row['Phases considered']
    full_path = main_folder #每个比例仅有一个结构，
    
    
    print('alloy: ', alloy)
    init_element, structure_type,init_element_lattice, target_size_angstrom = get_init_structure_param(elements, phase, target_size)
        
    initial_structure = build_target_size_model(init_element, structure_type,init_element_lattice, target_size_angstrom)#构建符合比例的初始模型

    for i in range(1):
        # 将模型进行随机替换
        structure = build_target_size_random_model(initial_structure, elements, fractions)        
        #save_structure(alloy, initial_structure, full_path, structure_type, i,'cif')  # 保存为CIF格式
        save_structure_for_one(alloy, initial_structure, full_path, structure_type, 'lammps-data')  # 保存为CIF格式
        #save_structure(alloy, initial_structure, full_path, 'cif')