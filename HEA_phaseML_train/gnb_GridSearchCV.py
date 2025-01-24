import pandas as pd
import os
import pickle
X_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_train.csv")
X_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_test.csv")
y_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_train.csv").values.ravel()
y_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_test.csv").values.ravel()




from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 定义超参数网格
var_smoothing = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # 合理范围的平滑参数

# 创建超参数网格
param_grid = {
    'var_smoothing': var_smoothing
}

# 使用 GridSearchCV 进行超参数搜索
gnb = GridSearchCV(
    estimator=GaussianNB(),
    param_grid=param_grid,
    cv=10,  # 交叉验证的折数
    n_jobs=-1,  # 使用所有可用的 CPU 核心
)

# 训练模型
GaussianNB_model = gnb.fit(X_train, np.array(y_train))

# 预测
train_pred = GaussianNB_model.predict(X_train)
test_pred = GaussianNB_model.predict(X_test)

# 计算性能指标
gnb_Accuracy_Train = accuracy_score(y_train, train_pred)
gnb_Accuracy_Test = accuracy_score(y_test, test_pred)

gnb_Precision_Train = precision_score(y_train, train_pred, average='weighted')
gnb_Precision_Test = precision_score(y_test, test_pred, average='weighted')

gnb_Recall_Train = recall_score(y_train, train_pred, average='weighted')
gnb_Recall_Test = recall_score(y_test, test_pred, average='weighted')

gnb_f1_score_Train = f1_score(y_train, train_pred, average='weighted')
gnb_f1_score_Test = f1_score(y_test, test_pred, average='weighted')

# 打印结果
print('gnb_Accuracy_Train', gnb_Accuracy_Train)
print('gnb_Accuracy_Test', gnb_Accuracy_Test)
print('gnb_Precision_Train', gnb_Precision_Train)
print('gnb_Precision_Test', gnb_Precision_Test)
print('gnb_Recall_Train', gnb_Recall_Train)
print('gnb_Recall_Test', gnb_Recall_Test)
print('gnb_f1_score_Train', gnb_f1_score_Train)
print('gnb_f1_score_Test', gnb_f1_score_Test)


model_name = 'GaussianNB_model'
os.makedirs(f'result/{model_name}', exist_ok=True)

# 保存结果到文本文件
with open(f'result/{model_name}/results.txt', 'w') as f:
    f.write(f'gnb_Accuracy_Train: {gnb_Accuracy_Train}\n')
    f.write(f'gnb_Accuracy_Test: {gnb_Accuracy_Test}\n')
    f.write(f'gnb_Precision_Train: {gnb_Precision_Train}\n')
    f.write(f'gnb_Precision_Test: {gnb_Precision_Test}\n')
    f.write(f'gnb_Recall_Train: {gnb_Recall_Train}\n')
    f.write(f'gnb_Recall_Test: {gnb_Recall_Test}\n')
    f.write(f'gnb_f1_score_Train: {gnb_f1_score_Train}\n')
    f.write(f'gnb_f1_score_Test: {gnb_f1_score_Test}\n')

# 保存最佳参数到文本文件
with open(f'result/{model_name}/best_parameters.txt', 'w') as f:
    f.write(f'Best Parameters: {gnb.best_params_}\n')

# 保存模型
with open(f'result/{model_name}/GaussianNB_model.pkl', 'wb') as f:
    pickle.dump(GaussianNB_model, f)