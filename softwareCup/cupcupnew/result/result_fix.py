import numpy as np
import pandas as pd

data=pd.read_csv('data_pred_submit_0825.csv')

data1=data.drop('Unnamed: 0',axis=1)

data2=data1.iloc[1:,:] # 去除第一行0 还剩44000011行，11天数据，每天第一条数据为0.

data2['day_id']=0

xxx=data2.pop('day_id')
data2.insert(0,'day_id',xxx)


for i in range(11):
	j=81+i
	data2.iloc[4000001*i:4000001*(i+1),0]=81+i

data3=data2[data2['2']!=0]			# 去 0

data3=data3.rename(columns={'0':'sale_nbr','1':'buy_nbr','2':'round'})

print('rename OK')


#data3.to_csv('data_pred_with_dayid_0830.csv')  #count: 2184490 

data4=data3
#for i in range(2184490):
#	data4.iloc[i,3]=int(data4.iloc[i,3])
print(data4.columns)
print(data4['round'])
data4['round']=data4['round']//10*10

data4.to_csv('data_pred_with_dayid_int_0830.csv')  #count: 2184490 
