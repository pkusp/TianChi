import numpy as np
import pandas as pd
from pandas import Series,DataFrame



fdata = pd.read_csv('power_sum.csv')
type(fdata.values)  # 2d array
power = fdata.values[:,1]  # column : power 

count=0  ## 全局变量用来读取数据
list_month=[]  ## 最终结果2d数组

for month_days in [31,28,31,30,31,30,31,31,30,31,30,31,31,29,31,30,31,30,31,31]:
	each_list=[]  ## 每月一个list
	for i in range(month_days):
		print(count)
		each_list.append(power[count])
		count=count+1
	list_len=len(each_list)  ## 判断该月数据量
	len_delta= 31-list_len   ## 不满31天则补零
	for i in range(len_delta):
		each_list.append(0)
	list_month.append(each_list)  ##  append单月的list

array_month=np.array(list_month)
df_month=DataFrame(array_month)

print(df_month)
print(type(df_month))

df_month.to_csv('power_mat31.csv')  ## 导出数据矩阵 mat_31_power.csv