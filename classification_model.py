
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import os

split_AUC = []

#the actual prediction model section (tested in a different file and it passes all 5 splits)
#Best Hyperparameters: {'C': 0.01, 'l1_ratio': 0.1}
#AUC Score: 0.98693
folder_name = f'F24_Proj3_data/split_1'
train_file = os.path.join(folder_name, 'train.csv')
test_file = os.path.join(folder_name, 'test.csv')
test_labels_file = os.path.join(folder_name, 'test_y.csv')

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
test_labels = pd.read_csv(test_labels_file)

X_train = train_data.iloc[:, 3:].values  
y_train = train_data['sentiment'].values
X_test = test_data.iloc[:, 2:].values  
y_test = test_labels['sentiment'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pred_model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=500, C=0.01, l1_ratio=0.1)
pred_model.fit(X_train, y_train)

y_pred_proba = pred_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
split_AUC.append(auc_score)