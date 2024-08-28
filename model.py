import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from autogluon.tabular import TabularDataset, TabularPredictor
import pickle

df_train = pd.read_csv(r'data\temp\trainset_new.csv')
df_val = pd.read_csv(r'data\temp\valset_new.csv')
df_test = pd.read_csv(r'data\temp\testset_new.csv')
# 加载数据集
x_train = df_train.drop(['shipID', 'UNLOCODE'], axis=1)
y_train = df_train['UNLOCODE']

x_val = df_val.drop(['shipID', 'UNLOCODE'], axis=1)
y_val = df_val['UNLOCODE']

x_test = df_test.drop(['INDEX', 'shipID'], axis=1)


feats = ['lon', 'lat', 'UNLOCODE']
LABEL = 'UNLOCODE'

# TabularDataset是一个继承pandas.DataFrame的数据集类
train_data = TabularDataset(pd.concat([df_train[feats], df_val[feats]], axis=0))
test_data = TabularDataset(x_test)
# 自动模型训练
predictor = TabularPredictor(problem_type='multiclass', label=LABEL).fit(train_data=train_data[feats])
# 模型预测
y_pred = predictor.predict(test_data)

with open(r'data\temp\label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
ans = encoder.inverse_transform(y_pred.values.reshape(-1, 1))
ans = pd.DataFrame(ans, columns=['UNLOCODE'])
datatest = pd.read_csv(r'data\dataTest.csv')
ans = pd.concat([df_test, ans], axis=1).set_index('INDEX', drop=True)
datatest.loc[ans.index, 'UNLOCODE'] = ans['UNLOCODE']
datatest.columns = datatest.columns.str.strip()
datatest['posTime'] = pd.to_datetime(datatest['posTime'], unit='s')
datatest.sort_values(['shipID', 'posTime'], inplace=True)
datatest['UNLOCODE'] = datatest['UNLOCODE'].fillna(method='bfill')
datatest.sort_values('INDEX', inplace=True)
datatest.to_csv(r'data\temp\result.csv', index=False)