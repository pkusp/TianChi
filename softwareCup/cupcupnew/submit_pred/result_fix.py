import numpy as np
import pandas as pd
from pandas import Series,DataFrame
'''
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

for j in range(11):
	for i in range(1225001):

		ok=cc.iloc[i,0]<12250001*j+1
		if ok:

while cc.iloc[i,0]<12250001:




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


'''
############# 20170904 #################################


cc=pd.read_csv('data_pred_submit_3500.csv')
bb=cc.rename(columns={'Unnamed: 0':'day_id','0':'sale_nbr','1':'buy_nbr','2':'round'})
bb['round']=bb['round']//100*100

aa=bb[bb['round']!=0]			# 去 0 lines:1625489


final_arr=['0','0','0','0']
for j in range(11):
	print(j)
	temp_df=aa[aa['day_id']<1225001*(j+1)+1]
	temp_df['day_id']=81+j

	temp_arr=np.array(temp_df)

	final_arr=np.vstack((final_arr,temp_arr))

final_arr=final_arr[1:,:]	
final_df1=DataFrame(final_arr)#,columns={'0':'day_id','1':'sale_nbr','2':'buy_nbr','3':'round'})

#final_df=final_df.rename(columns={'0':'day_id','1':'sale_nbr','2':'buy_nbr','3':'round'})
final_df2=final_df1.columns={'day_id','sale_nbr','buy_nbr','round'}

final_df3=final_df2[final_df2['day_id']!=0]			# 去 0 
final_df4=final_df3.iloc[1:,:]

final_df.to_csv('data_submit_with_dayid_0904.csv')




dd['cnt']=dd['round']/800
dd['cnt']=dd['cnt']//1
gg=dd[dd['round']>300]




















