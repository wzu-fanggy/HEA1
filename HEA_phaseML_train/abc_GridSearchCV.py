import pandas as pd


X_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_train.csv")
X_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_test.csv")
y_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_train.csv").values.ravel()
y_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_test.csv").values.ravel()


import os
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 定义超参数网格
param_grid = {
    'n_estimators': list(range(40, 150, 5)),
    'learning_rate': list(np.arange(1, 3, 0.25)),
    'algorithm': ['SAMME', 'SAMME.R'],
    'random_state': [0]
}

# 使用 GridSearchCV 进行超参数搜索
abc = GridSearchCV(
    estimator=AdaBoostClassifier(),
    param_grid=param_grid,
    cv=10,
    n_jobs=-1
)

# 训练模型
AdaBoost_model = abc.fit(X_train, np.array(y_train))

# 训练和测试预测
train_pred = AdaBoost_model.predict(X_train)
test_pred = AdaBoost_model.predict(X_test)

# 计算性能指标
abc_Accuracy_Train = accuracy_score(y_train, train_pred)
abc_Accuracy_Test = accuracy_score(y_test, test_pred)

abc_Precision_Train = precision_score(y_train, train_pred, average='weighted')
abc_Precision_Test = precision_score(y_test, test_pred, average='weighted')

abc_Recall_Train = recall_score(y_train, train_pred, average='weighted')
abc_Recall_Test = recall_score(y_test, test_pred, average='weighted')

abc_f1_score_Train = f1_score(y_train, train_pred, average='weighted')
abc_f1_score_Test = f1_score(y_test, test_pred, average='weighted')

# 输出性能指标
print('abc_Accuracy_Train:', abc_Accuracy_Train)
print('abc_Accuracy_Test:', abc_Accuracy_Test)
print('abc_Precision_Train:', abc_Precision_Train)
print('abc_Precision_Test:', abc_Precision_Test)
print('abc_Recall_Train:', abc_Recall_Train)
print('abc_Recall_Test:', abc_Recall_Test)
print('abc_f1_score_Train:', abc_f1_score_Train)
print('abc_f1_score_Test:', abc_f1_score_Test)

# 保存结果
model_name = 'ada_boost_model'
os.makedirs(f'result/{model_name}', exist_ok=True)

# 保存结果到文本文件
with open(f'result/{model_name}/results.txt', 'w') as f:
    f.write(f"Best Parameters: {abc.best_params_}\n")
    f.write(f"abc_Accuracy_Train: {abc_Accuracy_Train}\n")
    f.write(f"abc_Accuracy_Test: {abc_Accuracy_Test}\n")
    f.write(f"abc_Precision_Train: {abc_Precision_Train}\n")
    f.write(f"abc_Precision_Test: {abc_Precision_Test}\n")
    f.write(f"abc_Recall_Train: {abc_Recall_Train}\n")
    f.write(f"abc_Recall_Test: {abc_Recall_Test}\n")
    f.write(f"abc_f1_score_Train: {abc_f1_score_Train}\n")
    f.write(f"abc_f1_score_Test: {abc_f1_score_Test}\n")

# 保存最佳参数到文本文件
with open(f'result/{model_name}/best_parameters.txt', 'w') as f:
    f.write(f'Best Parameters: {abc.best_params_}\n')

# 保存模型
with open(f'result/{model_name}/ada_boost_model.pkl', 'wb') as f:
    pickle.dump(AdaBoost_model, f)