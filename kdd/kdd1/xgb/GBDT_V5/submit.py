#############################################
##
##   submit为合并数据成为提交结果的方法
##   题外话：有没有好的天气差值方法？？
##
############################################


import numpy as np
import pandas as pd

def loadData():

	j=0
	k=42

	listName=['A2','A3','B1','B3','C1','C3']

	for fileName in listName:

		fullName=fileName+'_morning_predict_0521_v5.csv'
		data=pd.read_csv(fullName)
		#print(data)
		data=data.drop('Unnamed: 0',axis=1) #去掉奇怪的一列
		#print(data)
		dataSeries=data['0']
		#print(dataSeries)
		dataList=list(dataSeries)
		#print(dataList)
		#print(dataList)  ### DataFrame一列
		#print(type(dataList))
		sample=pd.read_csv('submission_sample_travelTime.csv',index_col=0)
		#print(sample)
		#print(sample)
		#print(type(sample))
		sample['avg_travel_time'][j:k]=dataList
		#print(sample)
		j=j+42
		k=k+42
		sample.to_csv('submission_sample_travelTime.csv')


		#	for fileName in ['A2','A3','B1','B3','C1','C3']
	#listName=['A2','A3','B1','B3','C1','C3']
	for fileName in listName:

		fullName=fileName+'_afternoon_predict_0521_v5.csv'
		data=pd.read_csv(fullName)
		data=data.drop('Unnamed: 0',axis=1) #去掉奇怪的一列
		dataSeries=data['0']
		dataList=list(dataSeries)

		#print(dataList)  ### DataFrame一列
		#print(type(dataList))
		sample=pd.read_csv('submission_sample_travelTime.csv',index_col=0)
		#print(sample)
		#print(type(sample))
		sample['avg_travel_time'][j:k]=dataList
		j=j+42
		k=k+42
		sample.to_csv('submission_sample_travelTime.csv')





'''

for i in range(6):
	j=i*42


	sample=pd.read_csv('submission_sample_travelTime.csv')
	#print(sample)
	#print(type(sample))
	sample['avg_travel_time'][0:41]=dataList
	sample.to_csv('submission_sample_travelTime.csv')


'''