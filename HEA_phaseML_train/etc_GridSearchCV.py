import pandas as pd
import os
import pickle
X_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_train.csv")
X_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_X_test.csv")
y_train = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_train.csv").values.ravel()
y_test = pd.read_csv("../Datasets/state_Grid_smote/Classifier_y_test.csv").values.ravel()


import os
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 定义超参数网格
param_grid = {
    'n_estimators': list(range(50, 100, 5)),  # 减少估计器的数量
    'min_samples_split': list(range(3, 7)),  # 增加最小样本分裂数
    'max_depth': list(range(1, 10)),  # 降低最大深度
    'min_samples_leaf': list(range(2, 4)),  # 增加最小样本叶子数
    'max_features': ["sqrt", "log2"],
    'random_state': [0]
}

# 使用 GridSearchCV 进行超参数搜索
etc = GridSearchCV(
    estimator=ExtraTreesClassifier(), 
    param_grid=param_grid, 
    cv=10, 
    n_jobs=-1
)

# 训练模型
ExtraTreesClassifier_model = etc.fit(X_train, np.array(y_train))

# 训练和测试预测
train_pred = ExtraTreesClassifier_model.predict(X_train)
test_pred = ExtraTreesClassifier_model.predict(X_test)

# 计算性能指标
etc_Accuracy_Train = accuracy_score(y_train, train_pred)
etc_Accuracy_Test = accuracy_score(y_test, test_pred)

etc_Precision_Train = precision_score(y_train, train_pred, average='weighted')
etc_Precision_Test = precision_score(y_test, test_pred, average='weighted')

etc_Recall_Train = recall_score(y_train, train_pred, average='weighted')
etc_Recall_Test = recall_score(y_test, test_pred, average='weighted')

etc_f1_score_Train = f1_score(y_train, train_pred, average='weighted')
etc_f1_score_Test = f1_score(y_test, test_pred, average='weighted')

# 输出性能指标
print('etc_Accuracy_Train', etc_Accuracy_Train)
print('etc_Accuracy_Test', etc_Accuracy_Test)  # 修改为 etc_Accuracy_Test
print('etc_Precision_Train', etc_Precision_Train)
print('etc_Precision_Test', etc_Precision_Test)
print('etc_Recall_Train', etc_Recall_Train)
print('etc_Recall_Test', etc_Recall_Test)
print('etc_f1_score_Train', etc_f1_score_Train)
print('etc_f1_score_Test', etc_f1_score_Test)

# 保存结果
model_name = 'extra_trees_model'
os.makedirs(f'result/{model_name}', exist_ok=True)

# 保存结果到文本文件
with open(f'result/{model_name}/results.txt', 'w') as f:
    f.write(f"Best Parameters: {etc.best_params_}\n")
    f.write(f"etc_Accuracy_Train: {etc_Accuracy_Train}\n")
    f.write(f"etc_Accuracy_Test: {etc_Accuracy_Test}\n")
    f.write(f"etc_Precision_Train: {etc_Precision_Train}\n")
    f.write(f"etc_Precision_Test: {etc_Precision_Test}\n")
    f.write(f"etc_Recall_Train: {etc_Recall_Train}\n")
    f.write(f"etc_Recall_Test: {etc_Recall_Test}\n")
    f.write(f"etc_f1_score_Train: {etc_f1_score_Train}\n")
    f.write(f"etc_f1_score_Test: {etc_f1_score_Test}\n")

# 保存最佳参数到文本文件
with open(f'result/{model_name}/best_parameters.txt', 'w') as f:
    f.write(f'Best Parameters: {etc.best_params_}\n')

# 保存模型
with open(f'result/{model_name}/extra_trees_model.pkl', 'wb') as f:
    pickle.dump(ExtraTreesClassifier_model, f)