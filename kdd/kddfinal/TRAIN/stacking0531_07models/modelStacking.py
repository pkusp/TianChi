import pandas as pd
from sklearn.svm import SVR 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import skflow
'''
		train_x      //训练集           矩阵
		train_y     //训练集的label    向量

		test_x      //待预测的测试机    矩阵
'''

from sklearn.preprocessing import StandardScaler
######################################################  数据标准化  
ss_x=StandardScaler()
ss_y=StandardScaler()

train_x=ss_x.fit_transform(train_x)
train_y=ss_y.fit_transform(train_y)

test_x=ss_x.transform(test_x)
#######################################  还原方法 : inverse_transform  

def xgb_01(train_x,train_y,test_x):
	
	param={
		'booster':'gbtree',
		'objective': 'reg:linear', #多分类的问题
		#'num_class':10, # 类别数，与 multisoftmax 并用
		'n_estimators':100,
		#'reg_alpha':1,
		#'reg_lambda':200,

		'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
		'max_depth':6, # 构建树的深度，越大越容易过拟合
		'lambda':200,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
		'alpha':10,
		'subsample':0.7, # 随机采样训练样本
		'colsample_bytree':0.7, # 生成树时进行的列采样
		'min_child_weight':1, 
		# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
		#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
		#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
		'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
		'eta': 0.2, # 如同学习率
		'seed':500,
		#'nthread':7,# cpu 线程数
		#'eval_metric': 'auc'
		}

	print(i,' .. ')
	num_round=500
	#train_final=xgb.DMatrix(train_x,label=train_y)   ## 数据格式转换

		
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
		#xgb_cv_score=bst.cv(test_X_cv,test_Y_cv)  ####  此处应该有
		return each_mape

	#xgb_cv_score_02=xgb.cv(param, xg_train, num_round, nfold=5,metrics={'error'}, seed = 0,callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
	xgbMape=xgb_validation(train_x,train_y)
		
	#xgb_cv_score_02=xgb.cv(param, train_final, num_round, nfold=5,metrics={'error'}, seed = 0,callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
	test_final =xgb.DMatrix(test_x)
	#sumScore+=xgb_cv_score_02
	train_final=xgb.DMatrix(train_x,label=train_y)   ## 数据格式转换
	#watchlist=[(train_final,'train')]
	xgb_final=xgb.train(param,train_final,num_round)#,watchlist)

	pred_xgb=xgb_final.predict( test_final );
	#########################################################  xgb over #################
	#resultArr.append(pred_xgb)     ##   预测结果
	return pred_xgb

def gbdt_02(train_x,train_y,test_x):
	
	def gbdt_validation(x_name,label_name):                ### 验证函数
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
		gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！

		pred1=gbdt.predict(x_test)
		#print('          ########################      M    A     P   E  #############')			mapeErr=MAPE(y_test,pred1)
		gbdt_score=gbdt.score(x_test,y_test)
		return gbdt_score,mapeErr
		
	each_score,each_mape=gbdt_validation(train_x,train_y)  ### 每个时间窗口的误差
	sumScore+=abs(each_score)                    ### sumScore为全局变量
	mapeValue+=each_mape
	#print('\n')
	gbdt.fit(train_x,train_y) 

	pred_gbdt=gbdt.predict(test_x)   ##  预测结果，用test表示，
	############################################################################
	#resultArr.append(pred_gbdt)     ##   预测结果
	return pred_xgb

def svr_03(train_x,train_y,test_x):
	rbf_svr=SVR(kernel='rbf')
	##################  CV   ##########################
	rbf_cv=SVR(kernel='rbf ')
	train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
	rbf_cv.fit(train_X_cv,train_Y_cv)
	pred=rbf_cv.predict(test_X_cv)
	each_mape=MAPE(test_Y_cv,pred)    # cv 验证MAPE
	##################  CV   ##########################
	rbf_svr.fit(train_x,train_y)
	pred_svr=rbf_svr.predict(test_x)
	###########################################################################
	#resultArr.append(pred_svr)     ##   导出结果
	return pred_svr

def knr_04(train_x,train_y,test_x):
	dis_knr=KNeighborsRegressor(weights='distance')
	##################  CV   ##########################
	knr_cv=KNeighborsRegressor(weights='distance')
	train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
	knr_cv.fit(train_X_cv,train_Y_cv)
	pred=knr_cv.predict(test_X_cv)
	each_mape=MAPE(test_Y_cv,pred)    # cv 验证MAPE
	###################  CV   ###########################
	dis_knr.fit(train_x,train_y)
	pred_knr=dis_knr.predict(test_x)
	###########################################################################
	#resultArr.append(pred_knr)     ##   导出结果
	return pred_knr

def lr_05(train_x,train_y,test_x):
	lr=LinearRegression()
	##################  CV   ##########################
	lr_cv=LinearRegression()
	train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
	lr_cv.fit(train_X_cv,train_Y_cv)
	pred=lr_cv.predict(test_X_cv)
	each_mape=MAPE(test_Y_cv,pred)    # cv 验证MAPE
	###################  CV   ###########################
	lr.fit(train_x,train_y)
	pred_lr=lr.predict(test_x)
	###########################################################################
	#resultArr.append(pred_lr)     ##   导出结果
	return pred_lr

def rfr_06(train_x,train_y,test_x):
	rfr=RandomForestRegressor()
	##################  CV   ##########################
	rfr_cv=RandomForestRegressor()
	train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
	rfr_cv.fit(train_X_cv,train_Y_cv)
	pred=rfr_cv.predict(test_X_cv)
	each_mape=MAPE(test_Y_cv,pred)    # cv 验证MAPE
	###################  CV   ###########################
	rfr.fit(train_x,train_y)
	pred_rfr=rfr.predict(test_x)
	###########################################################################
	#resultArr.append(pred_rfr)     ##   导出结果
	return pred_rfr

def dnn_07(train_x,train_y,test_x):

			feature_columns = [tf.contrib.layers.real_valued_column("", dimension=18)]
			tf_dnn_regressor=skflow.DNNRegressor(feature_columns=feature_columns,hidden_units=[100,40],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),activation_fn=tf.nn.relu)
			##################  CV   ##########################
			dnn_cv = skflow.DNNRegressor(feature_columns=feature_columns,hidden_units=[100,40],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),activation_fn=tf.nn.relu)
			train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
			dnn_cv.fit(train_X_cv,train_Y_cv,batch_size=20,steps=10000)
			pred = dnn_cv.predict(test_X_cv)
			print('\n\n')
			print(pred)
			pred=list(pred)
			print(pred)
			print('\n\n\n\n')
			each_mape = MAPE(test_Y_cv,pred)    # cv 验证MAPE
			print(each_mape)
			###################  CV   ###########################

			tf_dnn_regressor.fit(train_x,train_y,batch_size=10,steps=1000)
			pred_dnn = tf_dnn_regressor.predict(test_x)
			pred_dnn = list(pred_dnn)
			pred_dnn = np.array(pred_dnn)
			###########################################################################
			#resultArr.append(pred_rfr)     ##   导出结果
			return pred_dnn

def arima08():
	pass

def lstm_09():
	pass






下面是一个对波士顿房屋价格的神经网络完整代码:
复制代码
 1 #  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 2 #
 3 #  Licensed under the Apache License, Version 2.0 (the "License");
 4 #  you may not use this file except in compliance with the License.
 5 #  You may obtain a copy of the License at
 6 #
 7 #   http://www.apache.org/licenses/LICENSE-2.0
 8 #
 9 #  Unless required by applicable law or agreed to in writing, software
10 #  distributed under the License is distributed on an "AS IS" BASIS,
11 #  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
12 #  See the License for the specific language governing permissions and
13 #  limitations under the License.
14 """DNNRegressor with custom input_fn for Housing dataset."""
15 
16 from __future__ import absolute_import
17 from __future__ import division
18 from __future__ import print_function
19 
20 import itertools
21 
22 import pandas as pd
23 import tensorflow as tf
24 # set logging verbosity to INFO 
25 tf.logging.set_verbosity(tf.logging.INFO)
26 #Define the column names for the data set in COLUMNS.To distinguish features from the label,also define FEATURES and LABEL.       # Then read the three CSVs into pandas DataFrame s:
27 COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
28            "dis", "tax", "ptratio", "medv"]
29 FEATURES = ["crim", "zn", "indus", "nox", "rm",
30             "age", "dis", "tax", "ptratio"]
31 LABEL = "medv"
32 
33 
34 def input_fn(data_set):
35   feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
36   labels = tf.constant(data_set[LABEL].values)
37   return feature_cols, labels
38 
39 
40 def main(unused_argv):
41   # Load datasets
42   training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
43                              skiprows=1, names=COLUMNS)
44   test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
45                          skiprows=1, names=COLUMNS)
46 
47   # Set of 6 examples for which to predict median house values
48   prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
49                                skiprows=1, names=COLUMNS)
50 
51   # Feature cols
52   feature_cols = [tf.contrib.layers.real_valued_column(k)
53                   for k in FEATURES]
54 
55   # Build 2 layer fully connected DNN with 10, 10 units respectively.
56   regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
57                                             hidden_units=[10, 10],
58                                             model_dir="/tmp/boston_model")
59 
60   # Fit
61   regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)
62 
63   # Score accuracy
64   ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
65   loss_score = ev["loss"]
66   print("Loss: {0:f}".format(loss_score))
67 
68   # Print out predictions
69   y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
70   # .predict() returns an iterator; convert to a list and print predictions
71   predictions = list(itertools.islice(y, 6))
72   print("Predictions: {}".format(str(predictions)))
73 
74 if __name__ == "__main__":
75   tf.app.run()
复制代码








feature_columns = [tf.contrib.layers.real_valued_column("", dimension=5)]
classifier = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
    hidden_units=[10],
    optimizer=tf.train.RMSPropOptimizer(learning_rate=.001),
    activation_fn=tf.nn.relu)
classifier.fit(x= train_data_input,
               y=train_data_outcomes,
               max_steps=1000)
print(classifier.evaluate(x= train_data_input, y=train_data_outcomes))







