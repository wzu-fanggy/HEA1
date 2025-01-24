import pandas as pd
import os
import numpy as np
import pickle
X_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_train.csv")
X_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_test.csv")
y_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_train.csv").values.ravel()
y_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_test.csv").values.ravel()

import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 超参数调整
param_grid = {
    'n_estimators': list(range(60, 180, 10)),  # 每隔10个数增加，减少计算量
    'min_samples_split': list(range(2, 8)),  # 增加最小样本分裂数
    'max_depth': list(range(8, 36, 2)),  # 每隔2个数增加，减少计算量
    'min_samples_leaf': list(range(2, 4)),  # 调整最小叶子节点
    'max_features': ["auto", "sqrt", "log2", None],
    'bootstrap': [True]
}

# 使用 GridSearchCV 进行超参数搜索
rfc = GridSearchCV(
    estimator=RandomForestClassifier(random_state=0),
    param_grid=param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1  # 使用所有可用的 CPU 核心
)

# 训练随机森林模型
RFC_model = rfc.fit(X_train, np.array(y_train))

# 输出最佳参数
best_params = RFC_model.best_params_
best_score = RFC_model.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)

# 模型评估
train_pred = RFC_model.predict(X_train)
test_pred = RFC_model.predict(X_test)

rfc_Accuracy_Train = accuracy_score(y_train, train_pred)
rfc_Accuracy_Test = accuracy_score(y_test, test_pred)

rfc_Precision_Train = precision_score(y_train, train_pred, average='weighted')
rfc_Precision_Test = precision_score(y_test, test_pred, average='weighted')

rfc_Recall_Train = recall_score(y_train, train_pred, average='weighted')
rfc_Recall_Test = recall_score(y_test, test_pred, average='weighted')

rfc_f1_score_Train = f1_score(y_train, train_pred, average='weighted')
rfc_f1_score_Test = f1_score(y_test, test_pred, average='weighted')

# 打印结果
print("rfc_Accuracy_train:", rfc_Accuracy_Train)
print("rfc_Accuracy_test:", rfc_Accuracy_Test)
print("rfc_Precision_train:", rfc_Precision_Train)
print("rfc_Precision_test:", rfc_Precision_Test)
print("rfc_Recall_train:", rfc_Recall_Train)
print("rfc_Recall_test:", rfc_Recall_Test)
print("rfc_f1_train:", rfc_f1_score_Train)
print("rfc_f1_test:", rfc_f1_score_Test)

# 保存结果
model_name = 'random_forest_model'
os.makedirs(f'result/{model_name}', exist_ok=True)

# 保存结果到文本文件
with open(f'result/{model_name}/results.txt', 'w') as f:
    f.write(f"Best Parameters: {best_params}\n")
    f.write(f"Best Cross-Validation Score: {best_score}\n")
    f.write(f"rfc_Accuracy_train: {rfc_Accuracy_Train}\n")
    f.write(f"rfc_Accuracy_test: {rfc_Accuracy_Test}\n")
    f.write(f"rfc_Precision_train: {rfc_Precision_Train}\n")
    f.write(f"rfc_Precision_test: {rfc_Precision_Test}\n")
    f.write(f"rfc_Recall_train: {rfc_Recall_Train}\n")
    f.write(f"rfc_Recall_test: {rfc_Recall_Test}\n")
    f.write(f"rfc_f1_train: {rfc_f1_score_Train}\n")
    f.write(f"rfc_f1_test: {rfc_f1_score_Test}\n")

# 保存最佳参数到文本文件
with open(f'result/{model_name}/best_parameters.txt', 'w') as f:
    f.write(f'Best Parameters: {best_params}\n')

# 保存模型
with open(f'result/{model_name}/random_forest_model.pkl', 'wb') as f:
    pickle.dump(RFC_model, f)