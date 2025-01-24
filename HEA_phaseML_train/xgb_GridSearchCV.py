import pandas as pd
import os
import pickle

X_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_train.csv")
X_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_test.csv")
y_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_train.csv").values.ravel()
y_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_test.csv").values.ravel()

import os
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 定义超参数网格
param_grid = {
    'booster': ['gbtree', 'dart'],
    'learning_rate': np.arange(0.01, 0.11, 0.01),
    'max_depth': np.arange(3, 6, 1),
    'subsample': np.arange(0.6, 0.91, 0.1),
    'colsample_bytree': np.arange(0.6, 0.81, 0.1),
    'lambda': np.arange(1, 3, 1),
    'alpha': np.arange(0.5, 2.61, 0.5),
    'n_estimators': list(range(30, 80, 5))
}

# 使用 GridSearchCV 进行参数搜索
xgbc = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    param_grid=param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 训练模型
XGB_Classifier_model = xgbc.fit(X_train, y_train)

# 最佳参数
best_params = xgbc.best_params_
print("最佳参数：", best_params)

# 训练和测试预测
train_pred = XGB_Classifier_model.predict(X_train)
test_pred = XGB_Classifier_model.predict(X_test)

# 计算性能指标
xgbc_Accuracy_Train = accuracy_score(y_train, train_pred)
xgbc_Accuracy_Test = accuracy_score(y_test, test_pred)

xgbc_Precision_Train = precision_score(y_train, train_pred, average='weighted')
xgbc_Precision_Test = precision_score(y_test, test_pred, average='weighted')

xgbc_Recall_Train = recall_score(y_train, train_pred, average='weighted')
xgbc_Recall_Test = recall_score(y_test, test_pred, average='weighted')

xgbc_f1_score_Train = f1_score(y_train, train_pred, average='weighted')
xgbc_f1_score_Test = f1_score(y_test, test_pred, average='weighted')

# 输出性能指标
print('xgbc_Accuracy_Train', xgbc_Accuracy_Train)
print('xgbc_Accuracy_Test', xgbc_Accuracy_Test)
print('xgbc_Precision_Train', xgbc_Precision_Train)
print('xgbc_Precision_Test', xgbc_Precision_Test)
print('xgbc_Recall_Train', xgbc_Recall_Train)
print('xgbc_Recall_Test', xgbc_Recall_Test)
print('xgbc_f1_score_Train', xgbc_f1_score_Train)
print('xgbc_f1_score_Test', xgbc_f1_score_Test)

# 保存结果
model_name = 'xgb_classifier_model'
os.makedirs(f'result/{model_name}', exist_ok=True)

# 保存结果到文本文件
with open(f'result/{model_name}/results.txt', 'w') as f:
    f.write(f"Best Parameters: {best_params}\n")
    f.write(f"xgbc_Accuracy_Train: {xgbc_Accuracy_Train}\n")
    f.write(f"xgbc_Accuracy_Test: {xgbc_Accuracy_Test}\n")
    f.write(f"xgbc_Precision_Train: {xgbc_Precision_Train}\n")
    f.write(f"xgbc_Precision_Test: {xgbc_Precision_Test}\n")
    f.write(f"xgbc_Recall_Train: {xgbc_Recall_Train}\n")
    f.write(f"xgbc_Recall_Test: {xgbc_Recall_Test}\n")
    f.write(f"xgbc_f1_score_Train: {xgbc_f1_score_Train}\n")
    f.write(f"xgbc_f1_score_Test: {xgbc_f1_score_Test}\n")

# 保存模型
with open(f'result/{model_name}/xgb_classifier_model.pkl', 'wb') as f:
    pickle.dump(XGB_Classifier_model, f)