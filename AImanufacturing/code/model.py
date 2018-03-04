#############################################
###
###  
##############################################
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.svm import SVR 
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from pandas import Series,DataFrame
#from sklearn.model_selection import train_test_split
#from sklearn import datasets
from sklearn import cross_validation
#import tensorflow as tf 
#import skflow
import xgboost as xgb
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier
#from sklearn.preprocessing import StandardScaler
#ss_x=StandardScaler()
#ss_y=StandardScaler()
import dataPre

num_round=10000
param={
	'booster':'gbtree',
	'objective': 'reg:gamma', 
	#'n_estimators':100,
	#'gamma':0.1, 
	'max_depth':3, 
	#'lambda':0.1, 
	#'alpha':10,
	'subsample':0.9, # 采样训练样本
	'colsample_bytree':0.5, # 列采样
	'min_child_weight':1, 
	'silent':1 ,
	'eta': 0.01, # 学习率
	#'seed':100,
	#'nthread':7,
	'eval_metric': 'mae'
	}

def MSE(yact,yhat):
	return ((yhat-yact)**2).mean()

def loadData(fileName):
	data=pd.read_csv(fileName)
	return data
def plotTwoLine(list_act,listRed):
	x=range(100)
	y1=list_act
	y2=listRed
	plt.plot(x,y1)
	plt.plot(x,y2,color='red')#,linestyle='--')
	plt.show()

def xgb_validation(x_name,label_name,param,num_round,random_num):   
	param=param
	num_round=num_round	
	train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=random_num)  # validation			
	print('train_X_cv:',train_X_cv.shape)
	print('train_Y_cv:',train_Y_cv.shape)
	print('test_X_cv:',test_X_cv.shape)
	print('test_Y_cva:',test_Y_cv.shape)

	xg_train = xgb.DMatrix( train_X_cv, label=train_Y_cv)
	xg_test = xgb.DMatrix(test_X_cv,label=test_Y_cv)
	watchlist = [(xg_train,'train'),(xg_test,'test')]
	bst = xgb.train(param,xg_train,num_round,watchlist,early_stopping_rounds=20)   ### TRAIN
	# get prediction
	pred_test = bst.predict( xg_test )
	cv_mse=MSE(test_Y_cv,pred_test)
	#plotTwoLine(test_Y_cv,pred_test)
	#xgb_cv_score=bst.cv(test_X_cv,test_Y_cv)  ####  此处应该有
	return cv_mse

def xgb_train(train_x,train_y,test_x,param,num_round):
	num_round=num_round
	param=param
	test_final =xgb.DMatrix(test_x)
	train_final=xgb.DMatrix(train_x,label=train_y)   ## 数据格式转换
	watchlist=[(train_final,'train')]
	xgb_final=xgb.train(param,train_final,num_round,watchlist,early_stopping_rounds=20)
	pred_xgb=xgb_final.predict( test_final )
	return pred_xgb #,yhat_xgb

def load_fertures(fileName,feature_size):
	n_d=feature_size # 降维后维度
	train_test_DF=loadData(fileName)
	train_test_arr=np.array(train_test_DF)
	train_x=train_test_arr[:500,1:n_d+1]
	train_y=train_test_arr[:500,n_d+1].reshape(500,1)#:n_d+2]
	test_x=train_test_arr[500:,1:n_d+1]
	#valid_x=train_test_arr[400:500,1:n_d+1]
	#valid_y=train_test_arr[400:500,n_d+1:n_d+2]
	return train_x,train_y,test_x

def gbdt(train_x,train_y,test_x,random_state):
	gbdt=GradientBoostingRegressor(             ####  上下午各一棵树
		loss='ls'                        #均方误差
		, learning_rate=0.01
		, n_estimators=50
		, subsample=0.9
		#, #min_samples_split=None
		#, #min_samples_leaf=None
		, max_depth=6
		, init=None
		#, #random_state=None
		#, #max_features=None
		, alpha=0.5
		, verbose=0
		#, #max_leaf_nodes=None
		, warm_start=False
		)

	def gbdt_validation(x_name,label_name):                ### 验证函数
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=random_state)  # validation			
		gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！
		pred=gbdt.predict(x_test)
		gbdt_mse=MSE(y_test,pred)
		print('\n\nMSE:\n',gbdt_mse)
		gbdt_score=gbdt.score(x_test,y_test)
		return gbdt_score,gbdt_mse
	
	each_score,mse=gbdt_validation(train_x,train_y) 
	print('each_score:',each_score)
	gbdt.fit(train_x,train_y) 
	pred_gbdt=gbdt.predict(test_x)  
	return pred_gbdt,mse



train_x,train_y,test_x=load_fertures('afterPCA_full_data.csv',500)
print('x:',train_x.shape)
print('y:',train_y.shape)
print('test:',test_x.shape)
'''
gbdt_mse_sum=0
for random_state in [100,202,15,10086,66,2018]:
	pred,gbdt_mse=gbdt(train_x,train_y,test_x,random_state)
	gbdt_mse_sum=gbdt_mse_sum+gbdt_mse
gbdt_mse_avg=gbdt_mse_sum/6

print('gbdt mse :',gbdt_mse_avg)

pred_gbdt,x=gbdt(train_x,train_y,test_x,666)
print(pred_gbdt)

'''
mse_sum=0
for random_num in [15,100,666,10086]:
	mse=xgb_validation(train_x,train_y,param,num_round,random_num)
	mse_sum=mse_sum+mse
mse_avg=mse_sum/4
print('MSE IS:',mse_avg)

'''
result=xgb_train(train_x,train_y,test_x,param,num_round)
print(result)
pred_id=pd.read_csv('../input/resultA.csv')
#pred_id=pred_id['id']
resultDF=DataFrame(result)
submission=pd.concat([pred_id,resultDF],axis=1)
submission.to_csv('../output/result_20180107.csv',header=False,index=False)
'''

