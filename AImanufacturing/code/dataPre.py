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
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import StandardScaler
######################################################  数据标准化  
#ss_x=StandardScaler()
#ss_y=StandardScaler()


def loadData(fileName):
	data=pd.read_csv(fileName)
	return data

dataFrame=loadData('train_test.csv')  # 500*8028

def num_missing(x):
    return sum(x.isnull())
def maxMinMedian(x):
    if not isinstance(x,str):
        return max(x),min(x)

print(dataFrame.apply(maxMinMedian,axis=0))
#print(dataFrame.apply(num_missing,axis=1))

'''
dataArr = np.array(dataFrame)
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
tempToolDF=DataFrame(tempArr)
tempToolDF.to_csv('allToolsRaw.csv') # 输出tools数据查看

print(tempArr.shape) 


####### the strategy is replace the string with int  ##########
for i in range(600):
    for j in range(13):
        if isinstance(tempArr[i,j],str):
            tempArr[i,j]=ord(tempArr[i,j][0])
         # string to int  # features

tempToolDF=DataFrame(tempArr)

tempToolDF.to_csv('allToolsInt.csv') # tools数据变为int后查看
#tempArr # 多出很多列。。？

#one hot encoder 
res=OneHotEncoder().fit(tempArr)
tempOneHotArr=res.transform(tempArr).toarray()
oneHotDF=DataFrame(tempOneHotArr)
oneHotDF.to_csv('oneHotTools.csv')

dataArr=np.array(dataFrame)
print(dataArr.shape)
print(dataArr[0,0])
dataArr=dataArr[:,1:] # ID duplicates
print(dataArr[0,0])

trainArr = np.hstack((tempOneHotArr,dataArr)) 

train_x=trainArr[:500,:-1]
train_y=trainArr[:500,-1].reshape(500,1)
test_x=trainArr[500:,:-1]  # 假设最后一列已被填充

print('tempArr:',tempArr.shape)
print('dataArr:',dataArr.shape)
print('trainArr:',trainArr.shape)
print('train_x:',train_x.shape)
print('trian_y:',train_y.shape)
print('test_x:',test_x.shape)

trainDF=DataFrame(trainArr)
#trainDF=trainDF.fillna(trainDF.mean()) # 均值填充空缺值
trainDF=trainDF.fillna(method='pad') # 上一个数据填充空缺值


trainArrForScale=np.array(trainDF)

def robust_scale(trainArr,arrLines):
    trainArrForScale=trainArr
    oneForth=np.percentile(trainArrForScale,10.0,axis=0)
    threeForth=np.percentile(trainArrForScale,90.0,axis=0)
    IQR=(threeForth-oneForth) # 四分位间距
    uu=np.percentile(trainArrForScale,100.0,axis=0)
    ll=np.percentile(trainArrForScale,0.0,axis=0)
    print('max:',uu)
    print('min:',ll)
    upperBound= threeForth+IQR
    lowerBound= oneForth-IQR
    print('upper:',upperBound)
    print('lower:',lowerBound)

    upperDF=upperBound- trainArrForScale 
    lowerDF= trainArrForScale - lowerBound

    for i in range(arrLines):
        for j in range(len(upperBound)):
            if upperDF[i,j]<0:
                #trainArrForScale[i,j]=upperBound[i]
                trainArrForScale[i,j]=trainArrForScale[i-1,j]
               
    for i in range(arrLines):
        for j in range(len(lowerBound)):
            if lowerDF[i,j]<0:
                #trainArrForScale[i,j]=lowerBound[i]
                trainArrForScale[i,j]=trainArrForScale[i-1,j]
    return trainArrForScale

robust_scaleARR= robust_scale(trainArrForScale,600)
robust_scale_DF=DataFrame(robust_scaleARR)
robust_scale_DF.to_csv('train_OneHot_padFill_robustScale.csv')
#trainDF.ffill()
trainDF.to_csv('trainWithOneHot_padFill.csv')

'''












