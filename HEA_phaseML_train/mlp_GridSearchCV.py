import pandas as pd
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 加载数据
X_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_train.csv")
X_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_test.csv")
y_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_train.csv").values.ravel()
y_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_test.csv").values.ravel()


# 定义超参数网格
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate_init': [0.001, 0.01],  # 调整学习率
    'max_iter': [500, 1000],  # 增加最大迭代次数
    'early_stopping': [True],  # 启用早停
    'validation_fraction': [0.1],  # 验证集比例
    'n_iter_no_change': [10],  # 没有变化的迭代次数
}

# 创建 MLPClassifier
mlp = MLPClassifier()

# 使用 GridSearchCV 进行超参数搜索
grid = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# 评估模型
train_pred = grid.predict(X_train)
test_pred = grid.predict(X_test)

# 计算指标
metrics = {
    'Accuracy_Train': accuracy_score(y_train, train_pred),
    'Accuracy_Test': accuracy_score(y_test, test_pred),
    'Precision_Train': precision_score(y_train, train_pred, average='micro'),
    'Precision_Test': precision_score(y_test, test_pred, average='micro'),
    'Recall_Train': recall_score(y_train, train_pred, average='micro'),
    'Recall_Test': recall_score(y_test, test_pred, average='micro'),
    'F1_Score_Train': f1_score(y_train, train_pred, average='micro'),
    'F1_Score_Test': f1_score(y_test, test_pred, average='micro'),
    'Best_Parameters': grid.best_params_
}

# 打印结果
for key, value in metrics.items():
    print(f'{key}: {value}')

# 保存结果和模型
model_name = 'mlp_classifier'
os.makedirs(f'result/{model_name}', exist_ok=True)

# 保存结果到文本文件
with open(f'result/{model_name}/results.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f'{key}: {value}\n')

# 保存最佳参数到文本文件
with open(f'result/{model_name}/best_parameters.txt', 'w') as f:
    f.write(f'Best Parameters: {grid.best_params_}\n')

# 保存模型
with open(f'result/{model_name}/mlp_model.pkl', 'wb') as f:
    pickle.dump(grid_result, f)