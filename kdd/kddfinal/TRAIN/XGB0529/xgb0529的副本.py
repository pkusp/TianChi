
#############################################

###  gbdt0520_v3.py 添加了weather数据，上午全部用09：00代替，下午全部用18：00代替
###  gbdt0521_v4.py 将时间窗口前移20min（train_x,train_y），使数据量翻倍
###
###  gbdt0521_v6.py 增加cross_validation，n次运行cross_validation保证误差稳定、
###  将上下午分开建树，采取不同的参数
###  增加MAPE函数（单条路）  
###  0523_v6.csv 修改n_estimators：100 to 20
###
###  0523_v2.py  增加特征 前一周同时间段（1*7），增加MAPE测评，n_estimators 为 6 时效果较好
###
###  gbdt0528.py  为kddfinal使用的代码，添加新数据，尝试修改loss functi in params of gbdt
###
###  xgb0529改自gbdt0528  numround=1000后才正常
###
##############################################
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from pandas import Series,DataFrame
#from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import cross_validation

import xgboost as xgb

#####################################################################################  XGB CV customize

print ('running cross validation, with cutomsized loss function')
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'error', float(sum(labels != (preds > 0.0))) / len(labels)

param = {'max_depth':2, 'eta':1, 'silent':1}
# train with customized objective
xgb.cv(param, dtrain, num_round, nfold = 5, seed = 0,
       obj = logregobj, feval=evalerror)
#####################################################################################  XGB CV customize

def MAPE(act,prd):
	err=abs(((act-prd)/act)).mean()
	return err

def runNtimes(fileName,n):   ## 多次run err取平均，使err稳定
	avgScore_1=0
	avgScore_2=0
	avgMape_1 =0
	avgMape_2 =0
	for i in range(n):
		sumScore_1,mapeValue_1=morning_xgb(fileName)   ### 调用each_gbdt 
		avgScore_1+=sumScore_1
		avgMape_1+=mapeValue_1
	for i in range(n):
		sumScore_2,mapeValue_2=afternoon_xgb(fileName)   ### 调用each_gbdt 
		avgScore_2+=sumScore_2
		avgMape_2+=mapeValue_2
	avgScore_1=avgScore_1/n
	avgScore_2=avgScore_2/n
	avgMape_1 =avgMape_1/n
	avgMape_2 =avgMape_2/n
	return avgScore_1,avgScore_2,avgMape_1,avgMape_2



def morning_xgb(fileName):     #return sumScore
	

	param={
	'booster':'gbtree',
	'objective': 'reg:linear', #多分类的问题
	#'num_class':10, # 类别数，与 multisoftmax 并用
	'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
	'max_depth':12, # 构建树的深度，越大越容易过拟合
	'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
	'subsample':0.7, # 随机采样训练样本
	'colsample_bytree':0.8, # 生成树时进行的列采样
	'min_child_weight':1, 
	# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
	#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
	#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
	'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
	'eta': 0.01, # 如同学习率
	'seed':1000,
	#'nthread':7,# cpu 线程数
	#'eval_metric': 'auc'
	}


	#######################################################   load data ！
	train_name = fileName+'_72matTrain.csv'
	test_name = fileName + '_swapTest__72mat.csv'
	train = pd.read_csv(train_name)                  ####### train data
	test = pd.read_csv(test_name)                    ####### test data

	train=train.drop('Unnamed: 0',axis=1)
	test=test.drop('Unnamed: 0',axis=1)

	train=np.array(train)    #DataFrame to ndarray
	test=np.array(test)

	###########################################################  morning    ！！！
	resultArr=[]
	sumScore=0
	mapeValue=0
	for i in range(6): 				### 六个时间窗口分别训练并验证

		j = 24+i

		train_old=train   ### 保存train数组，后续test数组需使用原数组10.05到10.17号的数据
		###############################################################################   TRAIN data 处理   ########################
		train_day_07=train[6:89,:72]    #七天之前的数组     ###### 以下分别求一周前的数组
		train_day_08=train[5:88,:72]    #8天之前的数组
		train_day_09=train[4:87,:72]    #9天之前的数组
		train_day_10=train[3:86,:72]    #10天之前的数组
		train_day_11=train[2:85,:72]    #11天之前的数组
		train_day_12=train[1:84,:72]    #12天之前的数组
		train_day_13=train[:83,:72]     #13天之前的数组

		train_76=train[14:,:72]          #############  train 切割为 76 行
		train_weather=train_old[14:,72:77]     #选中weather,要和train_76维度一样  ！！

		################################################  以下合并为大数组，包含八天数据 ######################
		train_time=np.hstack((train_76,train_day_07,train_day_08,train_day_09,train_day_10,train_day_11,train_day_12,train_day_13,))
		train_all_data=np.hstack((train_time,train_weather))   #大数组和weather整合,weather在最后   
		###################################################################################################################
		x00_train=train_all_data[:,18:24]       #当天6到8点          ####   以下提取特征 time（6+1*7）+weather（5）  ########
		xday07_train=train_all_data[:,(j+72):(j+72+1)]     #七天前本时间段 一列
		xday08_train=train_all_data[:,(j+72*2):(j+72*2+1)]   #八天前        一列!!!
		xday09_train=train_all_data[:,(j+72*3):(j+72*3+1)]	#九天
		xday10_train=train_all_data[:,(j+72*4):(j+72*4+1)]	#十天
		xday11_train=train_all_data[:,(j+72*5):(j+72*5+1)]	#十一天
		xday12_train=train_all_data[:,(j+72*6):(j+72*6+1)]	#十二天
		xday13_train=train_all_data[:,(j+72*7):(j+72*7+1)]	#十三天
		##################################################  以下特征合并为数组   ########################
		train_x_final=np.hstack((x00_train,xday07_train,xday08_train,xday09_train,xday10_train,xday11_train,xday12_train,xday13_train,train_weather))
		train_y_final=train_76[:,j]   #trian[]已经降为76行
		####################################################################
		train_x=np.mat(train_x_final)   ##转化为原名，方便后续函数调用
		train_y=train_y_final
		###############################################################################   TRAIN data 处理结束   ########################


		################################################################################   TEST data 处理  #########################
		#test_old=test[:,:72]                  ######### 以下分别求 一周前数组 test_old 为原test数组
		test_day07=train_old[90:,:72]   		##利用train[]内的数据  一周
		test_day08=train_old[89:96,:72]
		test_day09=train_old[88:95,:72]
		test_day10=train_old[87:94,:72]
		test_day11=train_old[86:93,:72]
		test_day12=train_old[85:92,:72]
		test_day13=train_old[84:91,:72]

		test_old=test[:,:72]  
		test_weather=test[:,72:77]
		################################################### 以下合并为大数组，包含8天数据
		test_time=np.hstack((test_old,test_day07,test_day08,test_day09,test_day10,test_day11,test_day12,test_day13,))
		test_all_data=np.hstack((test_time,test_weather))

		###################################################################################################################
		x00_test=test_all_data[:,18:24]       #当天6到8点          ####   以下提取特征 time（6+1*7）+weather（5）  ########
		xday07_test=test_all_data[:,j+72:j+72+1]     #七天前本时间段 一列
		xday08_test=test_all_data[:,j+72*2:j+72*2+1]   #八天前        一列
		xday09_test=test_all_data[:,j+72*3:j+72*3+1]	#九天
		xday10_test=test_all_data[:,j+72*4:j+72*4+1]	#十天
		xday11_test=test_all_data[:,j+72*5:j+72*5+1]	#十一天
		xday12_test=test_all_data[:,j+72*6:j+72*6+1]	#十二天
		xday13_test=test_all_data[:,j+72*7:j+72*7+1]	#十三天
		##################################################  以下特征合并为数组   ########################
		test_x_final=np.hstack((x00_test,xday07_test,xday08_test,xday09_test,xday10_test,xday11_test,xday12_test,xday13_test,test_weather))
		test_y_final=test_old[:,j]   #trian[]已经降为76行

		test_x=np.mat(test_x_final)
		################################################################################   TEST data 处理结束   ##########################

		###################################################################   XGB train   #####################################
		print('\n')
		num_round=1000
		train_final=xgb.DMatrix(train_x,label=train_y)   ## 数据格式转换

		xgb_cv_score_02=xgb.cv(param, train_final, num_round, nfold=5,metrics={'error'}, seed = 0,callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
		test_final =xgb.DMatrix(test_x)

		sumScore+=xgb_cv_score_02

		#watchlist=[(train_final,'train')]
		xgb_final=xgb.train(param,train_final,num_round)#,watchlist)
		pred_final=xgb_final.predict( test_final );
		#########################################################  xgb over #################
		resultArr.append(pred_final)     ##   预测结果
		

	resultArr = np.array(resultArr)  ###### 为了方便导出csv
	resultList=list(resultArr[:,0])+list(resultArr[:,1])+list(resultArr[:,2])+list(resultArr[:,3])+list(resultArr[:,4])+list(resultArr[:,5])+list(resultArr[:,6])
	

	###################################################### to csv
	resultDF=DataFrame(resultList)         ## mat to DF 
	resultDF.to_csv(fileName+'_AM_predict_xgb.csv')
	return sumScore,mapeValue

def afternoon_xgb(fileName):     #return sumScore
	

	param={
	'booster':'gbtree',
	'objective': 'reg:linear', #多分类的问题
	#'num_class':10, # 类别数，与 multisoftmax 并用
	'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
	'max_depth':12, # 构建树的深度，越大越容易过拟合
	'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
	'subsample':0.7, # 随机采样训练样本
	'colsample_bytree':0.8, # 生成树时进行的列采样
	'min_child_weight':1, 
	# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
	#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
	#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
	'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
	'eta': 0.01, # 如同学习率
	'seed':1000,
	'nthread':7,# cpu 线程数
	#'eval_metric': 'auc'
	}
	#######################################################   load data ！
	train_name = fileName+'_72matTrain.csv'
	test_name = fileName + '_swapTest__72mat.csv'
	train = pd.read_csv(train_name)                  ####### train data
	test = pd.read_csv(test_name)                    ####### test data

	train=train.drop('Unnamed: 0',axis=1)
	test=test.drop('Unnamed: 0',axis=1)

	train=np.array(train)    #DataFrame to ndarray
	test=np.array(test)

	###########################################################  morning    ！！！
	resultArr=[]
	sumScore=0
	mapeValue=0
	for i in range(6): 				### 六个时间窗口分别训练并验证

		#j = 24+i
		j = 51+i
		train_old=train   ### 保存train数组，后续test数组需使用原数组10.05到10.17号的数据
		###############################################################################   TRAIN data 处理   ########################
		train_day_07=train[6:89,:72]    #七天之前的数组     ###### 以下分别求一周前的数组
		train_day_08=train[5:88,:72]    #8天之前的数组
		train_day_09=train[4:87,:72]    #9天之前的数组
		train_day_10=train[3:86,:72]    #10天之前的数组
		train_day_11=train[2:85,:72]    #11天之前的数组
		train_day_12=train[1:84,:72]    #12天之前的数组
		train_day_13=train[:83,:72]     #13天之前的数组

		train_76=train[14:,:72]          #############  train 切割为 76 行
		#train_weather=train_old[14:,77:82]     #选中weather,要和train_76维度一样  ！！

		################################################  以下合并为大数组，包含八天数据 ######################
		train_time=np.hstack((train_76,train_day_07,train_day_08,train_day_09,train_day_10,train_day_11,train_day_12,train_day_13,))
		train_weather=train_old[14:,77:82]     #选中weather,要和train_76维度一样  ！！		
		train_all_data=np.hstack((train_time,train_weather))   #大数组和weather整合,weather在最后   
		###################################################################################################################
		x00_train=train_all_data[:,45:51]       #当天15到17点          ####   以下提取特征 time（6+1*7）+weather（5）  ########
		xday07_train=train_all_data[:,(j+72):(j+72+1)]      #七天前本时间段 一列
		xday08_train=train_all_data[:,(j+72*2):(j+72*2+1)]  #八天前        一列!!!
		xday09_train=train_all_data[:,(j+72*3):(j+72*3+1)]	#九天
		xday10_train=train_all_data[:,(j+72*4):(j+72*4+1)]	#十天
		xday11_train=train_all_data[:,(j+72*5):(j+72*5+1)]	#十一天
		xday12_train=train_all_data[:,(j+72*6):(j+72*6+1)]	#十二天
		xday13_train=train_all_data[:,(j+72*7):(j+72*7+1)]	#十三天
		##################################################  以下特征合并为数组   ########################
		train_x_final=np.hstack((x00_train,xday07_train,xday08_train,xday09_train,xday10_train,xday11_train,xday12_train,xday13_train,train_weather))
		train_y_final=train_76[:,j]   #trian[]已经降为76行
		####################################################################
		train_x=np.mat(train_x_final)   ##转化为原名，方便后续函数调用
		train_y=train_y_final
		###############################################################################   TRAIN data 处理结束   ########################


		################################################################################   TEST data 处理  #########################
		#test_old=test[:,:72]                  ######### 以下分别求 一周前数组 test_old 为原test数组
		test_day07=train_old[90:,:72]   		##利用train[]内的数据  一周
		test_day08=train_old[89:96,:72]
		test_day09=train_old[88:95,:72]
		test_day10=train_old[87:94,:72]
		test_day11=train_old[86:93,:72]
		test_day12=train_old[85:92,:72]
		test_day13=train_old[84:91,:72]

		test_old=test[:,:72]  
		#test_weather=test[:,77:82]
		################################################### 以下合并为大数组，包含8天数据
		test_time=np.hstack((test_old,test_day07,test_day08,test_day09,test_day10,test_day11,test_day12,test_day13,))
		test_weather=test[:,77:82]		
		test_all_data=np.hstack((test_time,test_weather))
		###################################################################################################################
		x00_test=test_all_data[:,45:51]       #当天6到8点          ####   以下提取特征 time（6+1*7）+weather（5）  ########
		xday07_test=test_all_data[:,j+72:j+72+1]     #七天前本时间段 一列
		xday08_test=test_all_data[:,j+72*2:j+72*2+1]   #八天前        一列
		xday09_test=test_all_data[:,j+72*3:j+72*3+1]	#九天
		xday10_test=test_all_data[:,j+72*4:j+72*4+1]	#十天
		xday11_test=test_all_data[:,j+72*5:j+72*5+1]	#十一天
		xday12_test=test_all_data[:,j+72*6:j+72*6+1]	#十二天
		xday13_test=test_all_data[:,j+72*7:j+72*7+1]	#十三天
		##################################################  以下特征合并为数组   ########################
		test_x_final=np.hstack((x00_test,xday07_test,xday08_test,xday09_test,xday10_test,xday11_test,xday12_test,xday13_test,test_weather))
		test_y_final=test_old[:,j]   #trian[]已经降为76行

		test_x=np.mat(test_x_final)
		################################################################################   TEST data 处理结束   ##########################

		###################################################################   XGB train   #####################################
		print('\n')
		num_round=1000
		train_final=xgb.DMatrix(train_x,label=train_y)   ## 数据格式转换

		xgb_cv_score_02=xgb.cv(param, train_final, num_round, nfold=5,metrics={'error'}, seed = 0,callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
		test_final =xgb.DMatrix(test_x)

		sumScore+=xgb_cv_score_02

		#watchlist=[(train_final,'train')]
		xgb_final=xgb.train(param,train_final,num_round)#,watchlist)
		pred_final=xgb_final.predict( test_final );
		#########################################################  xgb over #################
		resultArr.append(pred_final)     ##   预测结果


	resultArr = np.array(resultArr)  ###### 为了方便导出csv
	resultList=list(resultArr[:,0])+list(resultArr[:,1])+list(resultArr[:,2])+list(resultArr[:,3])+list(resultArr[:,4])+list(resultArr[:,5])+list(resultArr[:,6])


	###################################################### to csv
	resultDF=DataFrame(resultList)         ## mat to DF 
	resultDF.to_csv(fileName+'_PM_predict_xgb.csv')
	return sumScore,mapeValue









		'''
		def xgb_validation(x_name,label_name):   
			
			train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			

			xg_train = xgb.DMatrix( train_X_cv, label=train_Y_cv)
			xg_test = xgb.DMatrix(test_X_cv, label=test_Y_cv)


			watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
			num_round = 500
			bst = xgb.train(param, xg_train, num_round, watchlist );   ### TRAIN
			# get prediction
			pred = bst.predict( xg_test );
			xgb_cv_score=xgb.cv(param, xg_train, num_round, nfold=5,metrics={'error'}, seed = 0,callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

			#xgb_cv_score=bst.cv(test_X_cv,test_Y_cv)  ####  此处应该有
			return xgb_cv_score

		#xgb_cv_score_02=xgb.cv(param, xg_train, num_round, nfold=5,metrics={'error'}, seed = 0,callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

		xgbScore=xgb_validation(train_x,train_y)
		print(xgbScore)
		sumScore+=xgbScore
		'''




		'''
		def xgb_validation(x_name,label_name):   
			
			train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			

			xg_train = xgb.DMatrix( train_X_cv, label=train_Y_cv)
			xg_test = xgb.DMatrix(test_X_cv, label=test_Y_cv)


			watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
			num_round = 100
			bst = xgb.train(param, xg_train, num_round, watchlist );   ### TRAIN
			# get prediction
			pred = bst.predict( xg_test );
			xgb_cv_score=xgb.cv(param, xg_train, num_round, nfold=5,metrics={'error'}, seed = 0,callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

			#xgb_cv_score=bst.cv(test_X_cv,test_Y_cv)  ####  此处应该有
			return xgb_cv_score

		xgbScore=xgb_validation(train_x,train_y)
		print(xgbScore)
		sumScore+=xgbScore
		'''












