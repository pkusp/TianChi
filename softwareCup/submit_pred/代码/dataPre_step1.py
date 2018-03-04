import numpy as np
import pandas as pd
from pandas import Series,DataFrame


def read_data(fileName):
	data=pd.read_csv(fileName)

	sale_nbr=data['sale_nbr']            #选取sale列
	sales=sale_nbr.drop_duplicates()     #去重，
	sales=list(sales)
	#print(sales)

	round_list=[]
	round_list.append(sales)
	for day in range(91): # 外层循环，days
		day=day+1
		print('day ',day)
		each_day=data[data['day_id']==day]
		agent_list=[]
		for i in sales:  # 内层循环，agents
			#print('loading...  ',i,'  ',day)
			each_agent=each_day[each_day['sale_nbr']==i]
			each_agent_round=each_agent['round'].sum() #第day天第i个agent的销售额
			agent_list.append(each_agent_round)

		round_list.append(agent_list)

	return round_list #列表，每一行代表每一天，按agent排列的

round_list=read_data('sales_sample_20170310.csv')
print(round_list)
#rount_mat=np.mat(round_list)
round_DF=DataFrame(round_list)
round_DF.to_csv('round_mat_v1.csv')
