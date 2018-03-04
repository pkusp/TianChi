#
#
#   model_gbdt 特征过于简单粗暴，拟合的结果太平稳了，显然有问题
#
#
import numpy as np
import pandas as pd
#from sklearn.svm import SVR 
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from pandas import Series,DataFrame
#from sklearn.model_selection import train_test_split
#from sklearn import datasets
from sklearn import cross_validation
import tensorflow as tf 
import skflow
#import xgboost as xgb
#
# power_mat31_T:
#  0-------19
#  |
#  |
#  31
#

x=pd.read_csv('weather.csv')
y=pd.read_csv('power_sum.csv')

x=np.array(x)
y=np.array(y)

train_x=x[:-30,1:]
train_y=y[:,1:2]

test_x=x[-30:,1:]






def run(n):
	sum_score=0
	for i in range(n):
		sum_score+=gbdt_validation(train_x,train_y)
	return sum_score/n

gbdt=GradientBoostingRegressor(             ####  上下午各一棵树
	loss='ls'                        #均方误差
	, learning_rate=0.0001
	, n_estimators=100
	, subsample=0.8
	#, #min_samples_split=None
	#, #min_samples_leaf=None
	#, #max_depth=None
	, init=None
	#, #random_state=None
	#, #max_features=None
	, alpha=0.9
	, verbose=0
	#, #max_leaf_nodes=None
	, warm_start=False
	)

def value_fill(arr):
	for i in range(31):
		for j in range(20):
			if arr[i,j]==0:
				arr[i,j]=arr[i-1,j]  # 空缺值用前一天补齐

def data_pre(data):

	data_arr=np.array(data)
	value_fill(data_arr);        print(data_arr)

	train_1=data_arr[:,:13] 
	train_2=data_arr[:,1:14] 
	train_3=data_arr[:,2:15] 
	train_4=data_arr[:,3:16] 
	train_5=data_arr[:,4:17] 
	train_6=data_arr[:,5:18] 
	train_7=data_arr[:,6:19] 
	train_8=data_arr[:,7:20]

	train=np.vstack((train_1,train_2,train_3,train_4,train_5,train_6,train_7,train_8))

	test_x=data_arr[:,8:20] # test data
	train_x=train[:,:12]
	train_y=train[:,12]
	return train_x,train_y,test_x

def gbdt_validation(x_name,label_name):                ### 验证函数
	x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
	gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！
	#pred=gbdt.predict(x_test)
	gbdt_score=gbdt.score(x_test,y_test)
	return gbdt_score

def gbdt_train(train_x,train_y,test_x):
		
	each_score=gbdt_validation(train_x,train_y)  
	print('\n\n the score:')
	print(each_score)

	gbdt.fit(train_x,train_y) 
	pred_gbdt=gbdt.predict(test_x)   ##  预测结果，用test表示，
	##############################
	return pred_gbdt


def dnn(train_x,train_y,test_x):

	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
	tf_dnn_regressor=skflow.DNNRegressor(feature_columns=feature_columns,hidden_units=[20,20,20],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),activation_fn=tf.nn.relu)
	##################  CV   ##########################
	dnn_cv = skflow.DNNRegressor(feature_columns=feature_columns,hidden_units=[20,20,20],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),activation_fn=tf.nn.relu)
	train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
	dnn_cv.fit(x=train_X_cv,y=train_Y_cv,batch_size=10,steps=1000)
	pred = dnn_cv.predict(x=test_X_cv)
	print('\n\n')
	print('\n\n')
	print(pred)
	pred=np.array(pred)
	print(pred)
	print('\n\n\n\n')
	#each_mape = MAPE(test_Y_cv,pred)    # cv 验证MAPE
	#print(each_mape)
	###################  CV   ###########################

	tf_dnn_regressor.fit(x=train_x,y=train_y,batch_size=10,steps=1000)
	pred_dnn = tf_dnn_regressor.predict(x=test_x)
	pred_dnn = list(pred_dnn)
	pred_dnn = np.array(pred_dnn)
	#yhat_dnn=tf_dnn_regressor.predict(train_x)  ### yhat
	#yhat_dnn=list(yhat_dnn)
	#yhat_dnn=np.array(yhat_dnn)
	###########################################################################
	#resultArr.append(pred_rfr)     ##   导出结果
	return pred_dnn#,yhat_dnn









#data = pd.read_csv('power_mat31_T.csv')
#data = data.drop('Unnamed: 0',axis=1)

#train_x,train_y,test_x=data_pre(data)

#pred_label=gbdt_train(train_x,train_y,test_x)
pred_label=dnn(train_x,train_y,test_x) 
pred_result=pred_label[:-1]

print(pred_label)
'''
table=pd.read_csv('Tianchi_power_predict_table.csv')
table['predict_power_consumption']=pred_result

table.to_csv('Tianchi_power_predict_table.csv',index=None) ## index=none 防止多导出一列
'''




'''
def old_data():

	x_1=data_arr[:,12  ]; y_1=data_arr[:,12:13]
	x_2=data_arr[:,1:13]; y_1=data_arr[:,13:14]
	x_3=data_arr[:,2:14]; y_1=data_arr[:,14:15]
	x_4=data_arr[:,3:15]; y_1=data_arr[:,15:16]
	x_5=data_arr[:,4:16]; y_1=data_arr[:,16:17]
	x_6=data_arr[:,5:17]; y_1=data_arr[:,17:18]
	x_7=data_arr[:,6:18]; y_1=data_arr[:,18:19]
	x_8=data_arr[:,7:19]; y_1=data_arr[:,19:20]

	x_test=data_arr[:,8:20]  # 最后取完已无label可取
	y_test=[]
'''
