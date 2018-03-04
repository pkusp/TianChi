#############################################
### 20171219 将tools类别提取出来修改为int值，放在文件前13列，空缺值尚未填充
###  下一步：one hot encoding(tree model option), PCA降维，
##############################################
import pandas as pd
#from sklearn.svm import SVR 
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
import numpy as np
#from sklearn.ensemble import GradientBoostingRegressor
from pandas import Series,DataFrame
#from sklearn.model_selection import train_test_split
#from sklearn import datasets
#from sklearn import cross_validation
#import tensorflow as tf 
#import skflow
import xgboost as xgb
from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
######################################################  数据标准化  
#ss_x=StandardScaler()
#ss_y=StandardScaler()


def loadData(fileName):
	data=pd.read_csv(fileName)
	return data

dataFrame=loadData('train_test.csv')  # 500*8028
dataArr=np.array(dataFrame)
dataMatrix=np.mat(dataFrame)

print('dataMatrix:',dataMatrix.shape)

dataFrame.ffill() # 同列相邻数据填充
print('dataFrame:',dataFrame.shape)

tempList=[] # list to save tools
col=dataFrame.columns
for i in col:
    if i.startswith('T') or i.startswith('t'):
        print(i)
        #print(dataFrame[i]) # 打印所有tools
        tempList.append(dataFrame[i])  # TOOLs列append在一起
        dataFrame.drop([i],axis=1,inplace=True) # 原文件删除tools列，true为改变原文件

tempArr=np.array(tempList) # tools列array
tempArr=tempArr.T # 100*52, all tools
print(tempArr.shape)
####### the strategy is replace the string with int  ##########
for i in range(600):
    for j in range(13):
        if isinstance(tempArr[i,j],str):
            tempArr[i,j]=ord(tempArr[i,j][0])
         # string to int  # features

#tempArr # 多出很多列。。？

dataArr=np.array(dataFrame)
print(dataArr.shape)
print(dataArr[0,0])
dataArr=dataArr[:,1:] # ID duplicates
print(dataArr[0,0])

trainArr = np.hstack((tempArr,dataArr))

train_x=trainArr[:500,:8027]
train_y=trainArr[:500,8027:8028]
test_x=trainArr[500:,:8027]  # 假设最后一列已被填充

print('tempArr:',tempArr.shape)
print('dataArr:',dataArr.shape)
print('trainArr:',trainArr.shape)
print('train_x:',train_x.shape)
print('trian_y:',train_y.shape)
print('test_x:',test_x.shape)

trainDF=DataFrame(trainArr)
trainDF=trainDF.fillna(trainDF.mean()) # 均值填充空缺值
#trainDF.ffill()
trainDF.to_csv('trainPreWithInt_meanFill.csv')
















