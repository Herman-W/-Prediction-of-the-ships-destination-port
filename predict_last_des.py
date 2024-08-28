import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pickle
from minepy import MINE
import hashlib

def hash_encode(value):
    hash_object = hashlib.sha256(value.encode())
    hex_dig = hash_object.hexdigest()
    int_dig = int(hex_dig, 16)
    return int_dig% 1000000

def data_process(df, columns):
    for col in columns:
        df[col] = df[col].astype(str).apply(lambda x: hash_encode(x))
    return df

def make_label(df, df_label):
    df = pd.merge(df, df_label, on='shipID', how='left')
    df['load_in_berth_time'] = pd.to_datetime(df['load_in_berth_time'])
    df['discharge_in_berth_time'] = pd.to_datetime(df['discharge_in_berth_time'])
    df['posTime'] = pd.to_datetime(df['posTime'])
    df['eta'] = pd.to_datetime(df['eta'])
    #规则：上报时间在出发与到达时间之内
    df = df[(df['load_in_berth_time'] < df['posTime']) & (df['posTime'] < df['discharge_in_berth_time'])]
    df['time_diff'] = df['discharge_in_berth_time'] - df['load_in_berth_time']
    df = df.sort_values(by=['shipID', 'time_diff']).drop_duplicates(subset=['INDEX'], keep='first')
    return df

def mic(df, cols, target):
    x = df[cols].values
    y = data_process(df, target)[target].values.flatten()

    mic_values = pd.DataFrame(columns=cols)

    for i, feature in enumerate(cols):
        mine = MINE()
        mine.compute_score(x[:, i], y)
        mic = mine.mic()
        mic_values.loc[0, feature] = mic
    return mic_values


def label_encoder(df):
    encoder = OrdinalEncoder()
    df['UNLOCODE'] = encoder.fit_transform(df)
    with open(r'data\temp\label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    return df['UNLOCODE']

def label_encoder_load(df):    
    with open(r'data\temp\label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    df['UNLOCODE'] = encoder.transform(df)
    return df['UNLOCODE']


train_ais = pd.read_csv(r'data\dataTrainAIS.csv')
train_doc = pd.read_csv(r'data\dataTrainDoc.csv')
train_od = pd.read_csv(r'data\dataTrainOD.csv')

val_ais = pd.read_csv(r'data\dataValidAIS.csv')
val_doc = pd.read_csv(r'data\dataValidDoc.csv')
val_od = pd.read_csv(r'data\dataValidOD.csv')

test_ais = pd.read_csv(r'data\dataTest.csv')
test_doc = pd.read_csv(r'data\dataTestDoc.csv')
port = pd.read_csv(r'data\PortsData.csv')

train_ais.columns = train_ais.columns.str.strip()
val_ais.columns = val_ais.columns.str.strip()
test_ais.columns = test_ais.columns.str.strip()

train_ais['posTime'] = pd.to_datetime(train_ais['posTime'], unit='s')
val_ais['posTime'] = pd.to_datetime(val_ais['posTime'], unit='s')
test_ais['posTime'] = pd.to_datetime(test_ais['posTime'], unit='s')

train_ais.insert(0, 'INDEX', np.nan)
train_ais['INDEX'] = train_ais.index

val_ais.insert(0, 'INDEX', np.nan)
val_ais['INDEX'] = val_ais.index

train_od.drop_duplicates(inplace=True)
val_od.drop_duplicates(inplace=True)
#测试集划分航线
def assign_line(group):
    line = 0
    line_list = []
    group.reset_index(drop=True, inplace=True)
    for i, row in group.iterrows():
        if row['pos_diff'] > pd.Timedelta(days=5):
            line += 1
        line_list.append(line)
    line_df = pd.DataFrame({'line_id': line_list})
    group.reset_index(drop=True, inplace=True)
    group['line_id'] = line_df['line_id']
    lines = group.groupby('line_id').apply(lambda x: pd.DataFrame(x.iloc[-1]).T)
    return lines

test_ais['eta'] = pd.to_datetime(test_ais['eta'])
test_ais['posTime'] = pd.to_datetime(test_ais['posTime'])
test_ais.sort_values(['shipID', 'posTime'], inplace=True)
test_ais['eta_diff'] = test_ais.groupby('shipID')['eta'].diff()
test_ais['pos_diff'] = test_ais.groupby('shipID')['posTime'].diff()
test_line = test_ais.groupby('shipID').apply(assign_line)
test_line = test_line.drop(['load_port', 'posTime', 'cog', 'true_heading', 'sog', 'nav_status', 'eta', 'dest_port', 'breadth','length', 'eta_diff', 'pos_diff', 'line_id'], axis=1)
test_line = test_line.reset_index(drop=True)

#训练验证集制作标签与划分航线
train_od = pd.merge(train_od.rename(columns={'discharge_port': 'PortName'}), port, on='PortName', how='left').drop(['Country', 'CountryCode', 'CountryCodeISO2'], axis=1)
val_od = pd.merge(val_od.rename(columns={'discharge_port': 'PortName'}), port, on='PortName', how='left').drop(['Country', 'CountryCode', 'CountryCodeISO2'], axis=1)
trainset = make_label(train_ais, train_od)
train_line = trainset.sort_values(['shipID', 'posTime']).groupby('shipID').apply(lambda x: x.drop_duplicates(['load_in_berth_time','discharge_in_berth_time'], keep='last'))
train_line = train_line[(train_line['discharge_in_berth_time'] - train_line['posTime']) <= pd.Timedelta(days=2)].reset_index(drop=True)
train_line = train_line.drop(['INDEX', 'posTime', 'cog', 'true_heading','sog', 'nav_status', 'eta', 'dest_port', 'breadth', 'length','load_port', 'load_country', 'PortName', 'discharge_country','load_in_berth_time', 'discharge_in_berth_time', 'time_diff'], axis=1)

valset = make_label(val_ais, val_od)
val_line = valset.sort_values(['shipID', 'posTime']).groupby('shipID').apply(lambda x: x.drop_duplicates(['load_in_berth_time','discharge_in_berth_time'], keep='last'))
val_line = val_line.reset_index(drop=True)
val_line = val_line.drop(['INDEX', 'posTime', 'cog', 'true_heading','sog', 'nav_status', 'eta', 'dest_port', 'breadth', 'length','load_port', 'load_country', 'PortName', 'discharge_country','load_in_berth_time', 'discharge_in_berth_time', 'time_diff'], axis=1)

train_num = train_line.shape[0]
train_val_line = pd.concat([train_line, val_line])
#训练验证编码
train_val_line['UNLOCODE'] = label_encoder(train_val_line[['UNLOCODE']])
train_line = train_val_line[:train_num] 
val_line = train_val_line[train_num:]

train_line.to_csv(r'data\temp\trainset_new.csv', index=False)
val_line.to_csv(r'data\temp\valset_new.csv', index=False)
test_line.to_csv(r'data\temp\testset_new.csv', index=False)
print('==')