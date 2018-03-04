import numpy as np
import pandas as pd
from pandas import Series,DataFrame

def read_data(fileName):
	data=pd.read_csv(fileName)    # dataframe
	sale_nbr=data['sale_nbr']            #选取sale列，所有的 sales
	sales=sale_nbr.drop_duplicates()     #去重，
	sales=list(sales)

	buyer_nbr=data['buy_nbr']            # 所有的buyers
	buyers=buyer_nbr.drop_duplicates()     #去重，# 注意，PAX只剩一个了
	buyers=list(buyers)
	return sales,buyers

sales,buyers=read_data('sales_sample_20170310.csv')

sales=sales[:2000]		# 取2000个sale作为最终预测  ## 权宜之计，计算速度太慢 20170824
buyers=buyers[:2000]


def link_3d_pre(fileName):
	data=pd.read_csv(fileName)  # sale_to_buyer.csv
	## link file : sale_to_buyer table below
	##
	## 	        buyer 0 buyer 1 buyer 2 ... buyer n
	##  sale 0
	##  sale 1
	##  sale 2
	##
	##
	##  sale n 
	data_arr=np.array(data)
	#data_arr=data_arr[1:,1:] # 去掉名称行列
	#print(data_arr.shape)
	data_mat=np.mat(data_arr)
	data_mat=data_mat.T 
	#print(data_mat.shape)
	data_arr_t=np.array(data_mat)
	#          sale 0  sale 1  sale 2 ... sale n
	# buyer 0
	# buyer 1
	# 
	# 
	# buyer n

	data_arr_t=data_arr_t[2:2002,1:2001] # 去掉名称行列，取前2000


	#print('seee data_arr_t',data_arr_t[0,0])
	#print(data_arr_t.shape)
	sale_count=0
	count=0
	final_list=[[],[],[]] ############ 最终3行格式 sales buyer round
	final_arr=np.array(final_list)
	#print(final_arr)
	print(final_arr.shape)

	for i in sales:
		sale_count=sale_count+1
		print(sale_count)
		print(i)
		temp_list=[]  # 用来合并list
		each_sales_list=[] # 用来 each_sale*7000次

		for j in range(len(buyers)):
			each_sales_list.append(i) # i是each sale
		# ？？？count值有待验证！！！
		each_sale_buyers=data_arr_t[:,count] # 每对sale-buyer销售额 ndarray
		#all_buyers=data_arr_t[:,0]  # 所有buyers列 ndarray
		#print(each_sale_buyers)
		#for i in each_sale_buyers:

		each_sale_buyers=list(each_sale_buyers) # to list
		#all_buyers=list(all_buyers)  # to list
		all_buyers=buyers # list

		#print(all_buyers)
		#print(each_sale_buyers)
		#print(each_sales_list)

		temp_list.append(each_sales_list) # 重复7000+次each sale
		temp_list.append(all_buyers)      # 每次把所有 buyers列一次
		temp_list.append(each_sale_buyers) # 销售额对  ，以上共3d list

		temp_arr=np.array(temp_list)
		#print(temp_arr.shape)
		final_arr=np.hstack((final_arr,temp_arr)) # 这个要执行上千次，太蠢了！

		count=count+1 # 每次append下一行对所有buyer的销售额

	#   return final_arr:
	# sale0   sale0   sale0  ... salen  salen  salen
	# buyer0  buyer1  buyer2 ... buyer0 buyer1 buyer2
	# round   round   round      round  round  round
	return final_arr

final_arr=link_3d_pre('link_sale_to_buyer_both_name.csv')

final_mat=np.mat(final_arr)
final_mat=final_mat.T

final_df=DataFrame(final_mat)
final_df.to_csv('link_3d_pre.csv')






# next：将每行buyer较小的数去掉，剩下的求比例，然后将step 1 中预测的销售额按比例分配即可！！
# 困难点：剩下的数要整理好格式，最好直接和sale对应直接导出
# 等待 step 2 的数据，太慢了！


