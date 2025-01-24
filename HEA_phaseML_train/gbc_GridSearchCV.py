import pandas as pd
import os
import pickle
X_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_train.csv")
X_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_test.csv")
y_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_train.csv").values.ravel()
y_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_test.csv").values.ravel()



import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 定义超参数网格
param_grid = {
    'n_estimators': list(range(80, 165, 5)),
    'min_samples_split': list(range(2, 4)),
    'max_depth': list(range(2, 7)),
    'min_samples_leaf': list(range(1, 3)),
    'learning_rate': list(np.arange(0.01, 0.35, 0.05)),
    'random_state': [0]
}

# 使用 GridSearchCV 进行超参数搜索
gbc = GridSearchCV(
    estimator=GradientBoostingClassifier(), 
    param_grid=param_grid, 
    cv=10,
    n_jobs=-1
)

# 训练模型
gradientBoosting_model = gbc.fit(X_train, np.array(y_train))

# 训练和测试预测
train_pred = gradientBoosting_model.predict(X_train)
test_pred = gradientBoosting_model.predict(X_test)

# 计算性能指标
gbc_Accuracy_Train = accuracy_score(y_train, train_pred)
gbc_Accuracy_Test = accuracy_score(y_test, test_pred)

gbc_Precision_Train = precision_score(y_train, train_pred, average='weighted')
gbc_Precision_Test = precision_score(y_test, test_pred, average='weighted')

gbc_Recall_Train = recall_score(y_train, train_pred, average='weighted')
gbc_Recall_Test = recall_score(y_test, test_pred, average='weighted')

gbc_f1_score_Train = f1_score(y_train, train_pred, average='weighted')
gbc_f1_score_Test = f1_score(y_test, test_pred, average='weighted')

# 输出性能指标
print('gbc_Accuracy_Train:', gbc_Accuracy_Train)
print('gbc_Accuracy_Test:', gbc_Accuracy_Test)
print('gbc_Precision_Train:', gbc_Precision_Train)
print('gbc_Precision_Test:', gbc_Precision_Test)
print('gbc_Recall_Train:', gbc_Recall_Train)
print('gbc_Recall_Test:', gbc_Recall_Test)
print('gbc_f1_score_Train:', gbc_f1_score_Train)
print('gbc_f1_score_Test:', gbc_f1_score_Test)

# 保存结果
model_name = 'gradient_boosting_model'
os.makedirs(f'result/{model_name}', exist_ok=True)

# 保存结果到文本文件
with open(f'result/{model_name}/results.txt', 'w') as f:
    f.write(f"Best Parameters: {gbc.best_params_}\n")
    f.write(f"gbc_Accuracy_Train: {gbc_Accuracy_Train}\n")
    f.write(f"gbc_Accuracy_Test: {gbc_Accuracy_Test}\n")
    f.write(f"gbc_Precision_Train: {gbc_Precision_Train}\n")
    f.write(f"gbc_Precision_Test: {gbc_Precision_Test}\n")
    f.write(f"gbc_Recall_Train: {gbc_Recall_Train}\n")
    f.write(f"gbc_Recall_Test: {gbc_Recall_Test}\n")
    f.write(f"gbc_f1_score_Train: {gbc_f1_score_Train}\n")
    f.write(f"gbc_f1_score_Test: {gbc_f1_score_Test}\n")

# 保存最佳参数到文本文件
with open(f'result/{model_name}/best_parameters.txt', 'w') as f:
    f.write(f'Best Parameters: {gbc.best_params_}\n')

# 保存模型
with open(f'result/{model_name}/gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(gradientBoosting_model, f)