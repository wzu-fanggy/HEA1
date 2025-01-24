import pandas as pd

X_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_train.csv")
X_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_test.csv")
y_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_train.csv").values.ravel()
y_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_test.csv").values.ravel()

import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 定义超参数网格
leaf_size = list(range(20, 50, 5))  # 扩大范围以获得更多选择
n_neighbors = list(range(1, 21, 2))  # 增加邻居数范围
algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
p = [1, 2]  # 1表示曼哈顿距离，2表示欧几里得距离

# 创建超参数网格
param_grid = {
    'leaf_size': leaf_size,
    'n_neighbors': n_neighbors,
    'algorithm': algorithm,
    'p': p
}

# 使用 GridSearchCV 进行超参数搜索
knc = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=10,  # 交叉验证的折数
    n_jobs=-1,  # 使用所有可用的 CPU 核心
)

# 训练模型
kneighbors_model = knc.fit(X_train, np.array(y_train))

# KNeighbors Metrics
train_pred = kneighbors_model.predict(X_train)
test_pred = kneighbors_model.predict(X_test)

# 计算性能指标
knc_Accuracy_Train = accuracy_score(y_train, train_pred)
knc_Accuracy_Test = accuracy_score(y_test, test_pred)

# 指定average='macro'或'weighted'根据实际情况选择
knc_Precision_Train = precision_score(y_train, train_pred, average='weighted')
knc_Precision_Test = precision_score(y_test, test_pred, average='weighted')

knc_Recall_Train = recall_score(y_train, train_pred, average='weighted')
knc_Recall_Test = recall_score(y_test, test_pred, average='weighted')

knc_f1_score_Train = f1_score(y_train, train_pred, average='weighted')
knc_f1_score_Test = f1_score(y_test, test_pred, average='weighted')

# 打印结果
print('knc_Accuracy_Train', knc_Accuracy_Train)
print('knc_Accuracy_Test', knc_Accuracy_Test)
print('knc_Precision_Train', knc_Precision_Train)
print('knc_Precision_Test', knc_Precision_Test)
print('knc_Recall_Train', knc_Recall_Train)
print('knc_Recall_Test', knc_Recall_Test)
print('knc_f1_score_Train', knc_f1_score_Train)
print('knc_f1_score_Test', knc_f1_score_Test)

model_name = 'kneighbors_model'
os.makedirs(f'result/{model_name}', exist_ok=True)

# 保存结果到文本文件
with open(f'result/{model_name}/results.txt', 'w') as f:
    f.write(f'knc_Accuracy_Train: {knc_Accuracy_Train}\n')
    f.write(f'knc_Accuracy_Test: {knc_Accuracy_Test}\n')
    f.write(f'knc_Precision_Train: {knc_Precision_Train}\n')
    f.write(f'knc_Precision_Test: {knc_Precision_Test}\n')
    f.write(f'knc_Recall_Train: {knc_Recall_Train}\n')
    f.write(f'knc_Recall_Test: {knc_Recall_Test}\n')
    f.write(f'knc_f1_score_Train: {knc_f1_score_Train}\n')
    f.write(f'knc_f1_score_Test: {knc_f1_score_Test}\n')

# 保存最佳参数到文本文件
with open(f'result/{model_name}/best_parameters.txt', 'w') as f:
    f.write(f'Best Parameters: {knc.best_params_}\n')

# 保存模型
with open(f'result/{model_name}/kneighbors_model.pkl', 'wb') as f:
    pickle.dump(kneighbors_model, f)  # 修正为 kneighbors_model