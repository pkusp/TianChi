import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import time

start = time.clock()
middle = time.clock()

print (middle - start)


def read_data(fileName):
	data=pd.read_csv(fileName)    # dataframe
	sale_nbr=data['sale_nbr']            #选取sale列，所有的 sales
	sales=sale_nbr.drop_duplicates()     #去重，
	sales=list(sales)

	buyer_nbr=data['buy_nbr']            # 所有的buyers
	buyers=buyer_nbr.drop_duplicates()     #去重，
	buyers=list(buyers)
	return sales,buyers

#sales,buyers=read_data('sales_sample_20170310.csv')

#sales=sales[:2000]
#buyers=buyers[:2000]
def read_round_predict(fileName):
	data=pd.read_csv(fileName)
	data=data.drop('Unnamed: 0',axis=1)
	data_arr=np.array(data)
	data_arr_3500=data_arr[:3500,:]
	return data_arr_3500				# 只读前2000行sales

data_arr_3500=read_round_predict('pred_81_91_each_round_stackingResult.csv')

re_data_list_01=[0,0,0]
re_data_arr_01=np.array(re_data_list_01)

data_act_3d=pd.read_csv('link_3d_pre_3500.csv')
for i in range(11):
	print('day ',i)

	#re_data_list_02=[]
	#re_data_arr_02=np.array(re_data_list_02)

	re_data_list=[0,0,0]
	re_data_arr_02=np.array(re_data_list)

	data=data_arr_3500[:,i]  # 取出一列，即某一天的全部预测值

	#data_act=data_act_3d
	data_act=data_act_3d.drop('Unnamed: 0',axis=1)  # 每2000条为一个 sale 的 buyers
	data_arr=np.array(data_act)




	j=0 	# 每一个j对应一个 each_sale_round_pred 即某一天某个sale的销售额
	for each_sale_round_pred in data: # 一列中的每一个值 j 为一个sale的总销售额----预测值
		#print('sale round value ',each_sale_round_pred)
		#def date_duplicates(fileName):  # file is : link_3d_pre 2000*2000 条 列向

		##### 注意这的 drop 20170824
		#data_act=data_act_3d.drop('Unnamed: 0',axis=1)  # 每2000条为一个 sale 的 buyers

		#data_arr=np.array(data_act)
		# sale buyer round
		# sale
		# sale
		# sale
		#sales_act_sum_arr=[]   ？？？？？？？

		#re_data_list=[0,0,0]
		#re_data_arr_03=np.array(re_data_list)

		#for i in range(2000): 				# 每个占 2000行
		
		####################################################################
		print('compute number:',j)		# 这不是个真循环，而是为了顺次往下计算方便，每次2000行

		each_sale_arr=data_arr[3500*j:3500*(j+1),:] # 每一个sale提取出来
		#print(each_sale_arr)
		each_sale_round_sum=each_sale_arr[:,2].sum() #真实值，each的总额，用于下面求比例

		each_sale_arr[:,2]=each_sale_arr[:,2]/(1+each_sale_round_sum) # 将真实值变为每个buyer的占比

		#sales_act_sum_arr.append(each_sale_round_sum)    ？？？？？？？？？

		each_sale_arr[:,2]=each_sale_arr[:,2]*each_sale_round_pred # 将比例变为预测值
		#########################################################################
		j=j+1

		# vstack太蠢了，可否直接替换原arr(即data_arr)?
		#re_data_arr_03=np.vstack((re_data_arr_03,each_sale_arr)) # 每个2000行append在一起了

		re_data_arr_02=np.vstack((re_data_arr_02,each_sale_arr))  

	re_data_arr_01=np.vstack((re_data_arr_01,re_data_arr_02))


re_data_df=DataFrame(re_data_arr_01)
re_data_df=re_data_df[re_data_df['2']!=0]

re_data_df.to_csv('data_pred_submit_3500.csv')
	


	#data=data[data['2']!=0]


end = time.clock()
prit('/n/n/n/n/n/n/n')
print (end - start)
prit('/n/n/n/n/n/n/n')



