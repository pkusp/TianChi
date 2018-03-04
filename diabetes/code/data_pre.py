########
# 
# 20180105
# 
########
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler #四分位缩放
from sklearn.preprocessing import Normalizer # Normalizer(norm='l2').fit_transform(data)
from pandas import DataFrame

def num_missing(x):
    return sum(x.isnull())
def maxMinMedian(x):
    if not isinstance(x,str):
        return max(x),min(x)
def is_str(x):
	if isinstance(x,str):
		print(x)
		return x
#print(dataFrame.apply(is_str,axis=0))

train_df=pd.read_csv('../input/d_train.csv') # 5642*42
test_df=pd.read_csv('../input/d_test.csv')   # 1000*42
print('array shape :',np.array(train_df).shape)

''' 策略1：将乙肝相关稀疏数据直接丢弃,将检查日期丢弃 '''
train_df_1=train_df.drop(['id','date','ygky','ygkt','ygey','ygyt','yghx'],axis=1) # 去掉数据过于稀疏的几行（乙肝相关）5642*37
test_df_1=test_df.drop(['id','date','ygky','ygkt','ygey','ygyt','yghx'],axis=1) # 去掉数据过于稀疏的几行（乙肝相关）1000*37

''' 策略1.1：将异常值用0填充 '''
train_df_11=train_df_1.fillna(0)
test_df_11=test_df_1.fillna(0)

''' 策略1.2：数据标准化 '''
train_y=train_df_11['y']					# 提取y列，不参与标准化
train_df_11=train_df_11.drop(['y'],axis=1)  # drop train y列
test_df_11=test_df_11.drop(['y'],axis=1)    # drop test y 列

train_df_12 = DataFrame(MinMaxScaler().fit_transform(train_df_11))
test_df_12 = DataFrame(MinMaxScaler().fit_transform(test_df_11))
train_df_12['yact']=train_y # 恢复y列

train_df_12.to_csv('../tmp_file/train_norm.csv')
test_df_12.to_csv('../tmp_file/test_norm.csv')







