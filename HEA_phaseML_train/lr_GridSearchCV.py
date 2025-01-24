import pandas as pd
import os
import pickle

X_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_train.csv")
X_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_test.csv")
y_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_train.csv").values.ravel()
y_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_test.csv").values.ravel()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 定义超参数网格
penalty = ["l2", "elasticnet"]  # l1可能导致求解器失败
C = np.logspace(-4, 4, 10)  # 扩大C的范围
solver = ['lbfgs', 'saga']  # 适合多目标分类的求解器

# 创建超参数网格
param_grid = {
    'penalty': penalty,
    'C': C,
    'solver': solver,
    'max_iter': [100, 200, 300]  # 增加最大迭代次数
}

# 使用 GridSearchCV 进行超参数搜索
lr = GridSearchCV(
    estimator=LogisticRegression(multi_class='multinomial'),  # 设置为多类别
    param_grid=param_grid,
    cv=10,  # 交叉验证的折数
    n_jobs=-1,  # 使用所有可用的 CPU 核心
)

# 训练模型
lr_model = lr.fit(X_train, y_train)

# Logistic Regression Metrics
train_pred = lr_model.predict(X_train)
test_pred = lr_model.predict(X_test)

lr_Accuracy_Train = accuracy_score(np.array(y_train), train_pred)
lr_Accuracy_Test = accuracy_score(np.array(y_test), test_pred)
# 精确度（Precision）和召回率（Recall）
lr_Precision_Train = precision_score(np.array(y_train), train_pred, average='micro')
lr_Precision_Test = precision_score(np.array(y_test), test_pred, average='micro')

lr_Recall_Train = recall_score(np.array(y_train), train_pred, average='micro')
lr_Recall_Test = recall_score(np.array(y_test), test_pred, average='micro')

lr_f1_score_Train = f1_score(np.array(y_train), train_pred, average='micro')
lr_f1_score_Test = f1_score(np.array(y_test), test_pred, average='micro')

print('lr_Accuracy_Train', lr_Accuracy_Train)
print('lr_Accuracy_Test', lr_Accuracy_Test)
print('lr_Precision_Train', lr_Precision_Train)
print('lr_Precision_Test', lr_Precision_Test)
print('lr_Recall_Train', lr_Recall_Train)
print('lr_Recall_Test', lr_Recall_Test)
print('lr_f1_score_Train', lr_f1_score_Train)
print('lr_f1_score_Test', lr_f1_score_Test)

# 打印结果
results = {
    'Accuracy_Train': lr_Accuracy_Train,
    'Accuracy_Test': lr_Accuracy_Test,
    'Precision_Train': lr_Precision_Train,
    'Precision_Test': lr_Precision_Test,
    'Recall_Train': lr_Recall_Train,
    'Recall_Test': lr_Recall_Test,
    'F1_Score_Train': lr_f1_score_Train,
    'F1_Score_Test': lr_f1_score_Test,
    'Best_Parameters': lr.best_params_
}

for key, value in results.items():
    print(f'{key}: {value}')
# 保存结果和模型
# 确保结果文件夹存在，并包含模型名称
model_name = 'logistic_regression'
os.makedirs(f'result/{model_name}', exist_ok=True)

# 保存结果到文本文件
with open(f'result/{model_name}/results.txt', 'w') as f:
    for key, value in results.items():
        f.write(f'{key}: {value}\n')

# 保存最佳参数到文本文件
with open(f'result/{model_name}/best_parameters.txt', 'w') as f:
    f.write(f'Best Parameters: {lr.best_params_}\n')

# 保存模型
with open(f'result/{model_name}/lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)