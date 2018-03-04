
from pandas import Series,DataFrame

from numpy import *
import numpy as np

def loadDataSet(fileName):
	#从文件读出csv，并将string拆分为列表
	num=open(fileName).readline().strip().split(',')
	#原格式为string：'2016-07-20 00:00:00,10.6\n'
#In [61]: num
#Out[61]: ['2016-07-20 00:00:00', '10.6']

################# 将数据读成每天一条记录，20min一条数据，数据为avg time   #############
######  数据转化为ndarry类型   #########################################
	fr = open(fileName)
	dataMat=[]
	lineArr = []
	for line in fr.readlines():
		curLine = line.strip().split(',')  #列表出现
		lineArr.append(float(curLine[1]))  #append单个元素, curLine[1] is data of avg time
		if len(lineArr) == 24*3:           #每天72条数据     
			dataMat.append(lineArr)
			lineArr=[]

	new_name =fileName[0]+fileName[1]+'_72mat'+'01'
	ttt = new_name + '.csv'
	result = DataFrame(dataMat)
	result.to_csv(ttt)
    #########################    dataMat是list类型
    #### nddata为ndarray类型 ########################
	#dataMat.to_csv('A____mat.csv')                   #########  只有dataframe可以导出CSV  ##################
	print(ttt)
	print('new_hello')
	'''
	ddd = DataFrame(dataMat)
	ddd.to_csv('______72mat.csv')
'''
	#np.save("______mat.npy",dataMat)

	nddata = np.array(dataMat)       ################## ndata为24*3列,行数为天数 #########
	print(nddata)
	#nddata.to_csv('A____array.csv')
	#np.save("______array.npy",nddata)


	return nddata
    #print(nddata[0,:])      #####  共72列，从0：00：00 至 23：40：00  
    #nddata = np.insert(nddata, 0, 1, axis=1)  ##第0列插入值为1，即权重：W0  ######
    #### 在矩阵xMat中插入第0列，权重W0
	#xMat = np.mat( nddata[:,15:21] )         #### 最后一个值不取 #########
	#xarr = np.array(xMat)                    ##mat to arr

'''
	name =str(fileName)
	ttt = name + '.csv'
	ddd = DataFrame(dataMat)
	ddd.to_csv(ttt)

'''

