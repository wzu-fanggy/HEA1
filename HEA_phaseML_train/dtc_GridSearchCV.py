import pandas as pd
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_train.csv")
X_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_test.csv")
y_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_train.csv").values.ravel()
y_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_test.csv").values.ravel()

# 定义超参数网格
min_samples_split = list(range(2, 6))  # 增加范围以获得更多选择
max_depth = list(range(1, 15))
min_samples_leaf = list(range(1, 4))  # 扩大范围
max_features = ["sqrt", "log2"]  # 移除 'auto'

# 创建超参数网格
param_grid = {
    'min_samples_split': min_samples_split,
    'max_depth': max_depth,
    'min_samples_leaf': min_samples_leaf,
    'max_features': max_features,
}

# 使用 GridSearchCV 进行超参数搜索
dtc = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=param_grid,
    cv=10,  # 交叉验证的折数
    n_jobs=-1,  # 使用所有可用的 CPU 核心
)

# 训练模型
DTC_model = dtc.fit(X_train, np.array(y_train))

# 训练和测试预测
train_pred = DTC_model.predict(X_train)
test_pred = DTC_model.predict(X_test)

# 计算性能指标
dtc_Accuracy_Train = accuracy_score(y_train, train_pred)
dtc_Accuracy_Test = accuracy_score(y_test, test_pred)

# 对于多分类问题，设置average='weighted'以考虑类别不平衡
dtc_Precision_Train = precision_score(y_train, train_pred, average='weighted')
dtc_Precision_Test = precision_score(y_test, test_pred, average='weighted')

dtc_Recall_Train = recall_score(y_train, train_pred, average='weighted')
dtc_Recall_Test = recall_score(y_test, test_pred, average='weighted')

dtc_f1_score_Train = f1_score(y_train, train_pred, average='weighted')
dtc_f1_score_Test = f1_score(y_test, test_pred, average='weighted')

# 打印结果
print('dtc_Accuracy_Train:', dtc_Accuracy_Train)
print('dtc_Accuracy_Test:', dtc_Accuracy_Test)
print('dtc_Precision_Train:', dtc_Precision_Train)
print('dtc_Precision_Test:', dtc_Precision_Test)
print('dtc_Recall_Train:', dtc_Recall_Train)
print('dtc_Recall_Test:', dtc_Recall_Test)
print('dtc_f1_score_Train:', dtc_f1_score_Train)
print('dtc_f1_score_Test:', dtc_f1_score_Test)

model_name = 'decision_tree_model'
os.makedirs(f'result/{model_name}', exist_ok=True)

# 保存结果到文本文件
with open(f'result/{model_name}/results.txt', 'w') as f:
    f.write(f'dtc_Accuracy_Train: {dtc_Accuracy_Train}\n')
    f.write(f'dtc_Accuracy_Test: {dtc_Accuracy_Test}\n')
    f.write(f'dtc_Precision_Train: {dtc_Precision_Train}\n')
    f.write(f'dtc_Precision_Test: {dtc_Precision_Test}\n')
    f.write(f'dtc_Recall_Train: {dtc_Recall_Train}\n')
    f.write(f'dtc_Recall_Test: {dtc_Recall_Test}\n')
    f.write(f'dtc_f1_score_Train: {dtc_f1_score_Train}\n')
    f.write(f'dtc_f1_score_Test: {dtc_f1_score_Test}\n')

# 保存最佳参数到文本文件
with open(f'result/{model_name}/best_parameters.txt', 'w') as f:
    f.write(f'Best Parameters: {dtc.best_params_}\n')

# 保存模型
with open(f'result/{model_name}/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(DTC_model, f)  # 保存模型