# 七种元素：6468

import itertools
import json
import pandas as pd


# 定义元素列表
elements = ['Fe', 'Ni', 'Cr', 'Co', 'Cu', 'Al',  'Mg']

# 使用itertools.combinations生成所有可能的4个元素的组合
element_combinations_four = list(itertools.combinations(elements, 4))
print(len(element_combinations_four))

element_combinations_five = list(itertools.combinations(elements, 5))
print(len(element_combinations_five))

element_combinations_six = list(itertools.combinations(elements, 6))
print(len(element_combinations_six))


with open('dict_data_3_6_0.1.json', 'r') as f:
    loaded_dict_data = json.load(f)

por = loaded_dict_data['5']
pornum = len(loaded_dict_data['5'])
por1 = loaded_dict_data['5'][0]


compositions_all = []
proportions_four = loaded_dict_data['4'] # 4类 比例有多少个
proportions_five = loaded_dict_data['5'] # 5类 比例有多少个
proportions_six = loaded_dict_data['6'] # 6类 比例有多少个
compositions_all_formula = []


# 四元合金
for k in range(len(element_combinations_four)):
    elements = list(element_combinations_four[k])
    for i in range(len(proportions_four)):
        composition = {}
        tem_formula = ''
        for j in range(len(elements)):
            composition[elements[j]] = proportions_four[i][j]
            tem_formula += elements[j] +  "{:.2f}".format(proportions_four[i][j])  
        compositions_all_formula.append(tem_formula)
        compositions_all.append(composition)


# 五元合金
for k in range(len(element_combinations_five)):
    elements = list(element_combinations_five[k])
    for i in range(len(proportions_five)):
        composition = {}
        tem_formula = ''
        for j in range(len(elements)):
            composition[elements[j]] = proportions_five[i][j]
            tem_formula += elements[j] +  "{:.2f}".format(proportions_five[i][j])  
        compositions_all_formula.append(tem_formula)
        compositions_all.append(composition)

# 六元合金
for k in range(len(element_combinations_six)):
    elements = list(element_combinations_six[k])
    for i in range(len(proportions_six)):
        composition = {}
        tem_formula = ''
        for j in range(len(elements)):
            composition[elements[j]] = proportions_six[i][j]
            tem_formula += elements[j] +  "{:.2f}".format(proportions_six[i][j])  
        compositions_all_formula.append(tem_formula)
        compositions_all.append(composition)


#print(compositions_all)
#print(len(compositions_all))
print(compositions_all_formula)
print(len(compositions_all_formula))

df_test = pd.DataFrame(compositions_all)
df_test

# 我们将重新定义函数，以便它仅返回合金中存在的元素列表。
import re
def get_elements(alloy):
    # 使用正则表达式找到所有的元素
    pattern = re.compile(r'([A-Z][a-z]?)')
    # 解析合金字符串
    elements = pattern.findall(alloy)
    return elements

new_order = ['Alloy', 'Elements', 'Phases considered', 'number', 'Al', 'Co', 'Cr', 'Fe', 'Ni', 'Cu', 'Mn', 'Ti',
       'V', 'Nb', 'Mo', 'Zr', 'Hf', 'Ta', 'W', 'C', 'Mg', 'Zn', 'Si', 'N',
       'Li', 'Sn', 'B', 'Y', 'Pd','VEC', 'dSmix', 'Elect.Diff', 'Atom.Size.Diff', 'dHmix']
       
       
df_test = df_test.reindex(new_order, axis=1) 
df_test['Alloy'] = pd.DataFrame(compositions_all_formula)
df_test['Elements'] = df_test['Alloy'].apply(get_elements)
df_test['number'] = df_test['Elements'].apply(len)  

# 选择除了列'BPhases considered'之外的所有列，并将NaN值替换为0.0
cols = df_test.columns.difference(['Phases considered'])
df_test[cols] = df_test[cols].fillna(0.0)



xingzhi = pd.read_excel('Thermodynamic_properties.xlsx')


def VECCompute(df_composition):
    VEC =[]
    for i in range(len(df_composition)):
        temVEC = 0.0
        for el in df_composition['Elements'][i]:
            temVEC = temVEC + df_composition[el][i] * xingzhi[el][2]
        VEC.append(temVEC)
    return VEC

import math

def SmixCompute(df_composition):
    S =[]
    for i in range(len(df_composition)):
        temS = 0.0
        for el in df_composition['Elements'][i]:
            #print(df_new[el][i])
            if df_composition[el][i] ==  0.0:
                temS = temS + df_composition[el][i] * math.log(1e-10)
            else:
                temS = temS + df_composition[el][i] * math.log(df_composition[el][i])    
        temS = -8.3145*temS
        S.append(temS)
    return S


def ElectDiffCompute(df_composition):
    X =[]
    for i in range(len(df_composition)):
        temX = 0.0
        temAvgX = 0.0
        for el in df_composition['Elements'][i]:
            temAvgX = temAvgX + xingzhi[el][1]
        temAvgX = temAvgX/len(df_composition['Elements'][i])
        for el in df_composition['Elements'][i]:
            temX = temX + df_composition[el][i] * (xingzhi[el][1] - temAvgX)**2
        temX =  math.sqrt(temX)
        X.append(temX)
    return X


def AtomSizeDiffCompute(df_composition):
    D =[]
    for i in range(len(df_composition)):
        temD = 0.0
        temAvgD = 0.0
        for el in df_composition['Elements'][i]:
            temAvgD = temAvgD + xingzhi[el][0]
        temAvgD = temAvgD/len(df_composition['Elements'][i])
        for el in df_composition['Elements'][i]:
            temD = temD + df_composition[el][i] * (1 - xingzhi[el][0]/temAvgD)**2
        temD =  math.sqrt(temD)
        D.append(temD)
    return D

import numpy as np
import math


def listKey(lst):
    temlist = []
    for i in range(len(lst)):
        j = i + 1
        while j < len(lst):
            temlist.append(lst[i]+'-'+lst[j])
            j = j + 1
    return temlist

def H_compute(df_composition):
    Hij = pd.read_excel('Hij.xlsx', index_col=0, sheet_name = 0)
    Hij_dict = {}
    for column in Hij.columns:
        for index in Hij.index:
            if (column not in Hij_dict.keys()) and (index not in Hij_dict.keys()):
                temKey = column + '-' + index
                Hij_dict[temKey] = Hij[column][index]
    Hij1 = pd.read_excel('Hij.xlsx', index_col=0, sheet_name = 1)

    for column in Hij1.columns:
        for index in Hij1.index:
            if (column not in Hij_dict.keys()) and (index not in Hij_dict.keys()):
                if not (np.isnan(Hij1[column][index])):
                    temKey = column + '-' + index
                    Hij_dict[temKey] = Hij1[column][index]
    Hij2 = pd.read_excel('Hij.xlsx', index_col=0, sheet_name = 2)

    for column in Hij2.columns:
        for index in Hij2.index:
            if (column not in Hij_dict.keys()) and (index not in Hij_dict.keys()):
                if not (np.isnan(Hij2[column][index])):
                    temKey = column + '-' + index
                    Hij_dict[temKey] = Hij2[column][index]
    H =[]
    for i in range(len(df_composition)):
        temH = 0.0
        temListKey = listKey(df_composition['Elements'][i])
        for key in temListKey:
            elements = key.split('-')
            if key in Hij_dict.keys():
                temH = temH + 4 * Hij_dict[key] * df_composition[elements[0]][i] * df_composition[elements[1]][i]
            else:
                key1 = elements[1] + '-' + elements[0]
                if key1 in Hij_dict.keys():
                    temH = temH + 4 * Hij_dict[key1] * df_composition[elements[0]][i] * df_composition[elements[1]][i]
        H.append(temH)
    return H
    
    
VEC = VECCompute(df_test)
df_test['VEC'] = pd.DataFrame(VEC)    
    
S = SmixCompute(df_test)
df_test['dSmix'] = pd.DataFrame(S)
    
    
E = ElectDiffCompute(df_test)
df_test['Elect.Diff'] = pd.DataFrame(E)


D = AtomSizeDiffCompute(df_test)
df_test['Atom.Size.Diff'] = pd.DataFrame(D)


H = H_compute(df_test)
df_test['dHmix'] = pd.DataFrame(H)



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from imblearn.over_sampling import SMOTE


features =  ['Al', 'Co', 'Cr', 'Fe', 'Ni', 'Cu', 'Mn', 'Ti',
       'V', 'Nb', 'Mo', 'Zr', 'Hf', 'Ta', 'W', 'C', 'Mg', 'Zn', 'Si', 'N',
       'Li', 'Sn', 'B', 'Y', 'Pd','VEC', 'dSmix', 'Elect.Diff', 'Atom.Size.Diff', 'dHmix']

X = df_test[features]#, errors = 'ignore'.astype('float').values
y = df_test['Phases considered'].values

# 数据预处理和特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(len(X_scaled))
print(X_scaled[5])



import pickle

# 加载模型
# classifier_randomForestClassifier
# classifier_gradientBoosting
# classifier_XGB_Classifier
#  
#with open('Saved Models/Classifier Models/state_2/classifier_gradientBoosting.sav', 'rb') as file:
with open('gradient_boosting_model.pkl', 'rb') as file:
    model = pickle.load(file)

df_test['Phases considered'] = model.predict(X_scaled)



phase_counts = df_test['Phases considered'].value_counts()
print('phase_counts:', phase_counts)


df_test.to_csv("SevenHEA_all_proportion_Data.csv", index=False)