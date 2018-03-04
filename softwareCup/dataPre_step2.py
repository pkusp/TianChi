import numpy as np
import pandas as pd
from pandas import Series,DataFrame


def read_data(fileName):
	data=pd.read_csv(fileName)    # dataframe

	sale_nbr=data['sale_nbr']            #选取sale列，所有的 sales
	sales=sale_nbr.drop_duplicates()     #去重，
	sales=list(sales)

	buyer_nbr=data['buy_nbr']            # 所有的buyers
	buyers=buyer_nbr.drop_duplicates()     #去重，
	buyers=list(buyers)


	agent_list=[]
	agent_list.append(buyers) ##########  第一行append一排buyers

	for i in sales:  # 内层循环，agents
		print('sales:',i)
		each_agent=data[data['sale_nbr']==i]
		buyer_list=[]
		buyer_list.append(i)  # 每行第一个为 sales

		for j in buyers:  

			each_buyer=each_agent[each_agent['buy_nbr']==j] 
			each_bought=each_buyer['round'].sum() ########### 每对 sale to buyer 的总额！

			buyer_list.append(each_bought)

		agent_list.append(buyer_list)

	return agent_list



agent_list=read_data('sales_sample_20170310.csv')
agent_df=DataFrame(agent_list)

agent_df.to_csv('sale_to_buyer_link.csv')

'''


In [16]: agent_list_data=agent_list[1:]   ######### 第一排的buyers数量比下一行少一个，去掉

In [17]: agent_arr_data=np.array(agent_list_data)

In [18]: agent_array_data[1,1]


In [21]: df_withsale_link=DataFrame(agent_arr_data)

In [22]: df_withsale_link.to_csv('df_withsale_link.csv')

In [23]: type(agent_list_data)
Out[23]: list


In [21]: df_withsale_link=DataFrame(agent_arr_data)

In [22]: df_withsale_link.to_csv('df_withsale_link.csv')

In [23]: type(agent_list_data)
Out[23]: list

In [24]: type(buyers)


In [25]: data1=pd.read_csv('sales_sample_20170310.csv') 

In [26]: buyer_nbr1=data1['buy_nbr'] 

In [27]: buyers1=buyer_nbr1.drop_duplicates() 

In [28]: buyers1=list(buyers1)

In [29]: buyers1



In [30]: aaa=[0]

In [31]: len(buyers1)
Out[31]: 7372

In [32]: buyers2=aaa+buyers1

In [33]: len(buyers2)
Out[33]: 7373

In [34]: len(agent_list_data)
Out[34]: 7397

In [35]: len(agent_list_data[1])
Out[35]: 7373

In [36]: buyer_arr=np.array(buyers2)

In [37]: agent_array_data=np.array(agent_list_data)

In [38]: link_arr=np.vstack((buyer_arr,agent_array_data))

In [39]: link_df=DataFrame(link_arr)

In [40]: link_df.to_csv('link_sale_to_buyer_both_name.csv')

'''








