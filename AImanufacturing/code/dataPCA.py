#############################################
### 20171219 将tools类别提取出来修改为int值，放在文件前13列，空缺值尚未填充
###  下一步：one hot encoding(tree model option), PCA降维，
##############################################
import pandas as pd
from pandas import Series,DataFrame
from sklearn import preprocessing
import dataPre
from sklearn.decomposition import PCA


def replaceNanWithMean(matName):   # NAN value replace
    dataMat=matName
    numFeat=shape(dataMat)[1] # 0 为行，1为列
    for i in range(numFeat):
        meanVal=mean( dataMat[nonzero( ~isnan( dataMat[:,i] ))[0] ,i]    )
        dataMat[nonzero(isnan(dataMat[:,i]  ))[0],i]=meanVal
    return dataMat

#coding=utf-8
from numpy import *

'''通过方差的百分比来计算将数据降到多少维是比较合适的，
函数传入的参数是特征值和百分比percentage，返回需要降到的维度数num'''
def eigValPct(eigVals,percentage):
    sortArray=sort(eigVals) #使用numpy中的sort()对特征值按照从小到大排序
    sortArray=sortArray[-1::-1] #特征值从大到小排序
    arraySum=sum(sortArray) #数据全部的方差arraySum
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if tempSum>=arraySum*percentage:
            return num

'''pca函数有两个参数，其中dataMat是已经转换成矩阵matrix形式的数据集，列表示特征；
其中的percentage表示取前多少个特征需要达到的方差占比，默认为0.9'''
def pca00(dataMat,percentage=0.9):
    meanVals=mean(dataMat,axis=0)  #对每一列求平均值，因为协方差的计算中需要减去均值
    maxVals=max(dataMat)
    minVals=min(dataMat)
    print('meanVals:',meanVals)
    #meanRemoved=dataMat-meanVals
    meanRemoved=(dataMat-minVals)/(maxVals-minVals)
    print('meanRemoved:',meanRemoved)
    covMat=cov(meanRemoved,rowvar=0)  #cov()计算方差
    print('covMat1:',covMat)
    covMat = array(covMat, dtype=float) # dtype: object to float
    covMat=mat(covMat)
    print('covMat1:',covMat)
    covMat = nan_to_num(covMat) #用0填充 nan或inf
    print('covMat2:',covMat)
    eigVals,eigVects=linalg.eig(covMat)  #利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法   
    print('eigVals:',eigVals)
    k=eigValPct(eigVals,percentage) #要达到方差的百分比percentage，需要前k个向量
    eigValInd=argsort(eigVals)  #对特征值eigVals从小到大排序
    eigValInd=eigValInd[:-(k+1):-1] #从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    print('k:',k)
    redEigVects=eigVects[:,eigValInd]   #返回排序后特征值对应的特征向量redEigVects（主成分）
    print('redEigVects:',redEigVects)
    lowDDataMat=meanRemoved*redEigVects #将原始数据投影到主成分上得到新的低维数据lowDDataMat
    reconMat=(lowDDataMat*redEigVects.T)+meanVals   #得到重构数据reconMat
    return lowDDataMat,reconMat

def pca(dataMat,topNfeat=9999):
    meanVals=mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals
    covMat=cov(meanRemoved,rowvar=0)
    covMat = nan_to_num(covMat) #用0填充 nan或inf
    eigVals,eigVects=linalg.eig(mat(covMat))

    eigValInd=argsort(eigVals)
    eigValInd=eigValInd[-1:-(topNfeat+1):-1]

    redEigVects=eigVects[:,eigValInd]
    lowDDataMat=meanRemoved*redEigVects
    reconMat=(lowDDataMat*redEigVects.T)+meanVals
    return lowDDataMat,reconMat,covMat,eigVals,eigValInd,eigVects,redEigVects,meanRemoved,meanVals


#train_x_Test_mat=mat(train_xWithTest)
#fillDataMat=replaceNanWithMean(train_x_Test_mat)
#trainDF=dataPre.loadData('train_OneHot_padFill_robustScale.csv')
trainDF=dataPre.loadData('trainWithOneHot_padFill.csv')
trainArr=array(trainDF)
print(trainDF.shape)
print('trainArr shape:',trainArr.shape)
print('train:',trainArr[:,1])
train_xWithTest_arr=trainArr[:,1:-1] # 含有train和test的数据, 第一列为序号，去除
train_y=trainArr[:,-1].reshape(600,1)   # y


train_test_mat=mat(train_xWithTest_arr)

pca=PCA(n_components=500,whiten=True)
 
train_test_mat= nan_to_num(train_test_mat) #用0填充 nan或inf

lowD_data=pca.fit_transform(train_test_mat)
print(lowD_data)

lowD_fullArr=hstack((lowD_data,train_y))
print('full:',lowD_fullArr)

lowD_full_DF=DataFrame(lowD_fullArr)

lowD_full_DF.to_csv('afterPCA_full_data.csv')
#lowDDataMat,reconMat,covMat,eigVals,eigValInd,eigVects,redEigVects,meanRemoved,meanVals=pca(train_test_mat,50)

#lowDDataDF=DataFrame(lowDDataMat)
#reconDF=DataFrame(reconMat)

#lowDDataDF.to_csv('zeroMean_lowD_data.csv')

#reconDF.to_csv('recon_lowD_data.csv')

#lowDdataArr=array(lowDDataDF)
#zeroMeanLowDwithY=hstack((lowDdataArr,train_y))
#zeroMeanLowDwithYdf=DataFrame(zeroMeanLowDwithY)
#zeroMeanLowDwithYdf.to_csv('zeroMean_lowD_data_withY.csv')











