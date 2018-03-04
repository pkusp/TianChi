import numpy as np
import pandas as pd

data=pd.read_csv('data_pred_submit_0825.csv')

data1=data.drop('Unnamed: 0',axis=1)

data2=data1.iloc[1:,:] # 去除第一行0 还剩44000011行，11天数据，每天第一条数据为0.

data2['day_id']=0

xxx=data2.pop('day_id')
data2.insert(0,'day_id',xxx)

for i in range(44000011):			# 11天数据，81~91天
	data2.iloc[i,0]=i//400001+81	#将 day_id 写入每天的预测数据中

data3=data2[data2['2']!=0]			# 去 0

data3.to_csv('data_pred_with_dayid_0827.csv')


