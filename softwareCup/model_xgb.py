import numpy as np
import pandas as pd
#from sklearn.svm import SVR 
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import GradientBoostingRegressor
from pandas import Series,DataFrame
#from sklearn.model_selection import train_test_split
#from sklearn import datasets
from sklearn import cross_validation
#import tensorflow as tf 
#import skflow
import xgboost as xgb

#from sklearn.preprocessing import StandardScaler
######################################################  数据标准化  
#ss_x=StandardScaler()
#ss_y=StandardScaler()

def MAE(act,prd):
	err=abs(act-prd).mean()
	return err

def MAPE(act,prd):
	act=act+1
	err=abs((act-prd)/act).mean()
	return err

def data_pre(fileName):
	data_sale=pd.read_csv(fileName)
	#data_rank=pd.read_csv('fileName2')	#格式有问题
	data_sales=data_sale.drop('Unnamed: 0',axis=1)

	train_arr=np.array(data_sales) # train_arr为原始91列数据
	train_v=train_arr[:,1:51] #	取arr第一块作为初始训练集

	for i in range(2,31):		# 将所有训练集vstack
		train_each=train_arr[:,i:i+50]
		train_v=np.vstack((train_v,train_each))

	train_x=train_v[:,:49]
	train_y=train_v[:,49:50]
	print('data pre end...')
	return train_x,train_y,train_arr

train_x,train_y,train_arr=data_pre('round_mat_T_sales.csv')


param={
	'booster':'gbtree',
	'objective': 'reg:linear', 
	#'num_class':10, 
	'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守。
	'max_depth':6, # 构建树的深度，越大越容易过拟合
	'lambda':20,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
	'alpha':10,
	'subsample':0.7, # 随机采样训练样本
	'colsample_bytree':0.7, # 生成树时进行的列采样
	'min_child_weight':1, 
	
	'silent':1 ,
	'eta': 0.2, # 学习率
	'seed':500,
	#'nthread':7,
	#'eval_metric': 'auc'
	}

num_round=1000

'''
def xgb_validation(x_name,label_name):   

	train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			

	xg_train = xgb.DMatrix( train_X_cv, label=train_Y_cv)
	xg_test = xgb.DMatrix(test_X_cv)#, label=test_Y_cv)
	#watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
	#num_round = 500
	bst = xgb.train(param, xg_train, num_round)#, watchlist );   ### TRAIN
	# get prediction
	pred = bst.predict( xg_test )
	each_mape=MAPE(test_Y_cv,pred)
	print('each_mape')
	#xgb_cv_score=bst.cv(test_X_cv,test_Y_cv)  ####  此处应该有
	return each_mape

xgbMape=xgb_validation(train_x,train_y)
'''


train_final=xgb.DMatrix(train_x,label=train_y)   ## 数据格式转换
watchlist=[(train_final,'train')]
xgb_final=xgb.train(param,train_final,num_round,watchlist)

#train_x=xgb.DMatrix(train_x)
#########################################################  xgb #################


pred_list=[]
for i in range(81,92):
	print('predict number',i,' day')
	#print(train_arr)
	test_x=train_arr[:,i-49:i]
	test_final =xgb.DMatrix(test_x)

	pred_xgb_each_day=xgb_final.predict( test_final ) # 每一天预测所有代理人销售额
	pred_list.append(pred_xgb_each_day) # 将每天预测值append在一起

	MAPE_each_day=MAPE(train_arr[:,i],pred_xgb_each_day)
	MAE_each_day=MAE(train_arr[:,i],pred_xgb_each_day)

	print(MAPE_each_day)
	print(MAE_each_day)

#print(pred_list) # 最终预测结果


pred_df=DataFrame(pred_list)
pred_df=pred_df.T

#print(pred_df)
pred_df.to_csv('pred_81_91_each_round.csv')




