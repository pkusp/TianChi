#
#	将原始数据整理为 每个sales 每天的销售总额
#
import numpy as np
import pandas as pd
from pandas import Series,DataFrame


def read_data(fileName):
	data=pd.read_csv(fileName) # 文件名取为 data

	sale_nbr=data['sale_nbr']            # 选取sale列
	sales=sale_nbr.drop_duplicates()     # 去重，筛选出每个sale
	sales=list(sales)					 # 所有sale构成list
	#print(sales)

	round_list=[]
	round_list.append(sales)
	for day in range(91): # 外层循环，days # 每天一个循环，共90天
		day=day+1						 # 从第一天开始
		print('day ',day)				
		each_day =data[data['day_id']==day] # 将每天的数据单独取出，令其为 each_day
		agent_list=[]
		for i in sales:  # 内层循环，agents， 在所有 sale 的 list 里面依次处理
			#print('loading...  ',i,'  ',day)
			each_agent=each_day[each_day['sale_nbr']==i] # 将每个sale的所有数据取出，令其为 each_agent
			each_agent_round=each_agent['round'].sum()	#将每day的每个agent的金额sum(), each_agent_round
			agent_list.append(each_agent_round)			# 每个agent 的 sum数值 append进 每day的 agent_list

		round_list.append(agent_list)		# 将每day的list append进总list

	return round_list #列表，每一行代表每一天，按agent排列的

round_list=read_data('sales_sample_20170310.csv')
print(round_list)
#rount_mat=np.mat(round_list)
round_DF=DataFrame(round_list)
round_DF.to_csv('round_mat_v1.csv') # 90天的每个sale的销售总额，横轴为 sales，纵轴为天数



