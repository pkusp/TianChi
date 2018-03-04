'''
from __future__ import print_function
from __future__ import division

from abc import ABCMeta
from abc import abstractmethod

from .base import BaseEnsemble
from ..base import BaseEstimator
from ..base import ClassifierMixin
from ..base import RegressorMixin
from ..externals import six
from ..feature_selection.from_model import _LearntSelectorMixin

from ._gradient_boosting import predict_stages
from ._gradient_boosting import predict_stage
from ._gradient_boosting import _random_sample_mask

import numbers
import numpy as np

from scipy import stats
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from time import time
from ..tree.tree import DecisionTreeRegressor
from ..tree._tree import DTYPE
from ..tree._tree import TREE_LEAF

from ..utils import check_random_state 
from ..utils import check_array
from ..utils import check_X_y
from ..utils import column_or_1d
from ..utils import check_consistent_length
from ..utils import deprecated
from ..utils.extmath import logsumexp
from ..utils.fixes import expit
from ..utils.fixes import bincount
from ..utils.stats import _weighted_percentile
from ..utils.validation import check_is_fitted
from ..utils.validation import  NotFittedError
from ..utils.multiclass import check_classification_targets
'''
'''
import six
from abc import ABCMeta
from abc import abstractmethod
'''
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
##############################################
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from pandas import Series,DataFrame
#from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import cross_validation



'''

class LossFunction(six.with_metaclass(ABCMeta, object)):
    """Abstract base class for various loss functions.

    Attributes
    ----------
    K : int
        The number of regression trees to be induced;
        1 for regression and binary classification;
        ``n_classes`` for multi-class classification.
    """

    is_multi_class = False

    def __init__(self, n_classes):
        self.K = n_classes

    def init_estimator(self):
        """Default ``init`` estimator for loss function. """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, y, pred, sample_weight=None):
        """Compute the loss of prediction ``pred`` and ``y``. """

    @abstractmethod
    def negative_gradient(self, y, y_pred, **kargs):
        """Compute the negative gradient.

        Parameters
        ---------
        y : np.ndarray, shape=(n,)
            The target labels.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_weight, sample_mask,
                                learning_rate=1.0, k=0):
        """Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray, shape=(n, m)
            The data array.
        y : ndarray, shape=(n,)
            The target labels.
        residual : ndarray, shape=(n,)
            The residuals (usually the negative gradient).
        y_pred : ndarray, shape=(n,)
            The predictions.
        sample_weight : ndarray, shape=(n,)
            The weight of each sample.
        sample_mask : ndarray, shape=(n,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default 0
            The index of the estimator being updated.

        """
        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # update each leaf (= perform line search)
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions,
                                         leaf, X, y, residual,
                                         y_pred[:, k], sample_weight)

        # update predictions (both in-bag and out-of-bag)
        y_pred[:, k] += (learning_rate
                         * tree.value[:, 0, 0].take(terminal_regions, axis=0))

    @abstractmethod
    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """Template method for updating terminal regions (=leaves). """

class RegressionLossFunction(six.with_metaclass(ABCMeta, LossFunction)):
    """Base class for regression loss functions. """

    def __init__(self, n_classes):
        if n_classes != 1:
            raise ValueError("``n_classes`` must be 1 for regression but "
                             "was %r" % n_classes)
        super(RegressionLossFunction, self).__init__(n_classes)

class MapeError(RegressionLossFunction):                   ###  /Users/sp/anaconda3/anaconda/lib/python3.5/site-packages/sklearn/ensemble/gradient_boosting ##line273
    """Loss function for least squares (LS) estimation.
    Terminal regions need not to be updated for least squares. """
    def init_estimator(self):
        return MeanEstimator()

    def __call__(self, y, pred, sample_weight=None):
        if sample_weight is None:
            return np.mean( ((y - pred.ravel())/y ) ** 2.0)    #### 增加了 /y  mape表示方法
        else:
            return (1.0 / sample_weight.sum() *
                    np.sum(sample_weight * ( ((y - pred.ravel())/y ) ** 2.0)))

    def negative_gradient(self, y, pred, **kargs):
        return (y - pred.ravel())/y

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_weight, sample_mask,
                                learning_rate=1.0, k=0):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        y_pred[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        pass

LOSS_FUNCTIONS = {#'ls': LeastSquaresError,
                  #'lad': LeastAbsoluteError,
                  #'huber': HuberLossFunction,
                  #'quantile': QuantileLossFunction,
                  #'deviance': None,    # for both, multinomial and binomial
                  #'exponential': ExponentialLoss,
                  'MAPE':MapeError,                ### 我自定义的ERROR！！！
                  }
'''



def MAPE(act,prd):
	err=abs(((act-prd)/act)).mean()
	return err
'''
def gbdt_validation(x_name,label_name):                ### 验证函数
	x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
	gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！
			
	pred1=gbdt.predict(x_test)
	print('          ########################      M    A     P   E  #############')
	print(MAPE(y_test,pred1))
			
	gbdt_score=gbdt.score(x_test,y_test)
	return gbdt_score
'''


def runNtimes(fileName,n):   ## 多次run err取平均，使err稳定
	avgScore_1=0
	avgScore_2=0
	avgMape_1 =0
	avgMape_2 =0
	for i in range(n):
		sumScore_1,mapeValue_1=morning_gbdt(fileName)   ### 调用each_gbdt 
		avgScore_1+=sumScore_1
		avgMape_1+=mapeValue_1
	for i in range(n):
		sumScore_2,mapeValue_2=afternoon_gbdt(fileName)   ### 调用each_gbdt 
		avgScore_2+=sumScore_2
		avgMape_2+=mapeValue_2
	avgScore_1=avgScore_1/n
	avgScore_2=avgScore_2/n
	avgMape_1 =avgMape_1/n
	avgMape_2 =avgMape_2/n
	return avgScore_1,avgScore_2#,avgMape_1,avgMape_2

def morning_gbdt(fileName):     #return sumScore
	
	gbdt=GradientBoostingRegressor(             ####  上下午各一棵树
	loss='ls'                        # MAPE误差                   # line 653 source file
	, learning_rate=0.005
	, n_estimators=20
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

		
		def gbdt_validation(x_name,label_name):                ### 验证函数
			x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
			gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！
			
			pred1=gbdt.predict(x_test)
			#print('          ########################      M    A     P   E  #############')
			mapeErr=MAPE(y_test,pred1)
			#print(mapeErr)
			
			gbdt_score=gbdt.score(x_test,y_test)
			return gbdt_score,mapeErr
		
		each_score,each_mape=gbdt_validation(train_x,train_y)  ### 每个时间窗口的误差
		sumScore+=abs(each_score)                    ### sumScore为全局变量
		mapeValue+=each_mape
		#print('\n')
		#print(fileName + '第 '+ str(float(i)) + '个时间窗口的 平方误差 为：：')
		#print(each_score)
		#print('\n')
		gbdt.fit(train_x,train_y) 
		pred=gbdt.predict(test_x)   ##  预测结果，用test表示，
		resultArr.append(pred)     ##   预测结果

	#print('\n')
	#print(fileName+'的morning总误差为：')
	#print(sumScore)
	#print(fileName+'的morninig的 ####  MAPE #### 为：：：')
	#print(mapeValue/6)
	#resultMat=np.mat(resultArr)        # arr to mat
	#resultMat=resultMat.T           # .T
	resultArr = np.array(resultArr)  ###### 为了方便导出csv
	resultList=list(resultArr[:,0])+list(resultArr[:,1])+list(resultArr[:,2])+list(resultArr[:,3])+list(resultArr[:,4])+list(resultArr[:,5])+list(resultArr[:,6])
	
	#print('\n')
	#print('this result is: '+fileName+'new morning data prediction!!')
	#print('\n')

	###################################################### to csv
	resultDF=DataFrame(resultList)         ## mat to DF 
	#resultDF.to_csv(fileName+'_AM_predict.csv')
	return sumScore#,mapeValue

def afternoon_gbdt(fileName):     #return sumScore
	
	gbdt=GradientBoostingRegressor(             ####  上下午各一棵树
	loss='ls'                        #均方误差
	, learning_rate=0.005
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

		
		def gbdt_validation(x_name,label_name):                ### 验证函数
			x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
			gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！
			
			pred1=gbdt.predict(x_test)
			#print('          ########################      M    A     P   E  下午 #############')
			mapeErr=MAPE(y_test,pred1)
			#print(mapeErr)

			gbdt_score=gbdt.score(x_test,y_test)
			return gbdt_score,mapeErr
		
		each_score,each_mape=gbdt_validation(train_x,train_y)  ### 每个时间窗口的误差
		sumScore+=abs(each_score)                    ### sumScore为全局变量
		mapeValue+=each_mape
		#print('\n')
		#print(fileName + '第 '+ str(float(i)) + '个时间窗口的 平方误差 为：：')
		#print(each_score)
		#print('\n')
		gbdt.fit(train_x,train_y) 
		pred=gbdt.predict(test_x)   ##  预测结果，用test表示，
		resultArr.append(pred)     ##   预测结果

	#print('\n')
	#print(fileName+'的afternoo总误差为：')
	#print(sumScore)
	#print(fileName+'的morninig的 ####  MAPE #### 为：：：')
	#print(mapeValue/6)	
	#resultMat=np.mat(resultArr)        # arr to mat
	#resultMat=resultMat.T           # .T
	resultArr = np.array(resultArr)  ###### 为了方便导出csv
	resultList=list(resultArr[:,0])+list(resultArr[:,1])+list(resultArr[:,2])+list(resultArr[:,3])+list(resultArr[:,4])+list(resultArr[:,5])+list(resultArr[:,6])
	
	#print('\n')
	#print('this result is: '+fileName+'new afternoon data prediction!!')
	#print('\n')

	###################################################### to csv
	resultDF=DataFrame(resultList)         ## mat to DF 
	#resultDF.to_csv(fileName+'_PM_predict.csv')
	return sumScore,mapeValue









'''
def afternoon_gbdt(fileName):

	gbdt=GradientBoostingRegressor(             ####  上下午各一棵树
	loss='ls'                        #均方误差
	, learning_rate=0.1
	, n_estimators=20
	, subsample=0.6
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


	train_name = fileName+'_72mat.csv'
	test_name = fileName + '_72mat01.csv'
	train = pd.read_csv(train_name)
	test = pd.read_csv(test_name)

	train=train.drop('Unnamed: 0',axis=1)   ### 读出的csv系统会自动添加索引，故drop it
	test=test.drop('Unnamed: 0',axis=1)

	train=np.array(train)    #DataFrame to ndarray
	test=np.array(test)

	###########################################################  afternoon    ！！！
	resultArr=[]
	#resultList=[]
	sumScore=0
	for i in range(6): 				### 六个时间窗口分别训练并验证
		j = 51+i
		#train_x01=train[:,45:51]   #特征 15:00 to 17:00
		#train_x02=train[:,77:82]   #天气特征
		#train_x = np.hstack((train_x01,train_x02))  #合并特征数组
		#train_y=train[:,j]        #label 从第51列开始，共执行六次
		
		train_x01=train[:,45:51]   #时间特征
		train_x02=train[:,77:82]   #天气特征   #位于第72：76列

		train_x03=train[:,44:50]
		train_x04=train[:,77:82]   #天气特征   #位于第72：76列

		train_x05=train[:,46:52]
		train_x06=train[:,77:82]   #天气特征   #位于第72：76列	

		train_x_1 = np.hstack((train_x01,train_x02))  #合并特征数组 横向增加特征值
		train_x_2 = np.hstack((train_x03,train_x04))  #合并特征数组
		train_x_3 = np.hstack((train_x05,train_x06))  #合并特征数组

		train_y1=train[:,j]        #label 从第24列开始，共执行六次
		train_y2=train[:,j-1]
		train_y3=train[:,j+1]

		train_x=np.vstack((train_x_1,train_x_2,train_x_3)) #合并特征数组 纵向增加数据量
		train_y=np.hstack((train_y1,train_y2,train_y3))

		test_x01=test[:,45:51]     #测试数据
		test_x02=test[:,77:82]    #天气特征
		test_x = np.hstack((test_x01,test_x02))  #合并特征数组
		#test_id=test[24]
		#print(test_x)
		#print('\n\n\n')
		train_x=np.mat(train_x)   # ndarray to mat
		test_x =np.mat(test_x)

		#train_y=np.mat(train_y).T    ## 列向量！！！！！！！！！  ## 竟然不要列向量！！
		#print(train_feat.shape)#,train_id.shape,test_feat.shape)#,test_id.shape)

		#x_train, x_test, y_train, y_test = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation
		#gbdt.fit(x_train,y_train)  ####  GBDT训练在这开始 ！！
		#gbdt_score=gbdt.score(x_test,y_test)
		def gbdt_validation(x_name,label_name):                ### 验证函数
			x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
			gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！
			pred2=gbdt.predict(x_test)

			print('          ########################      M    A     P   E  #############')
			print(MAPE(y_test,pred2))
			
			gbdt_score=gbdt.score(x_test,y_test)
			return gbdt_score

		each_score=gbdt_validation(train_x,train_y)  ### 每个时间窗口的误差
		sumScore+=abs(each_score)                    ### sumScore为全局变量

		#print('\n')
		print(fileName + '第 '+ str(float(i)) + '个时间窗口的 平方误差 为：：')
		print(each_score)
		#print('\n')

		gbdt.fit(train_x,train_y) 
		pred=gbdt.predict(test_x)   ##  预测结果，用test表示， test_x 在此处用到 ！！！
		resultArr.append(pred)     ##   预测结果

	#print('\n')
	print(fileName+'的afternoon总误差为：')
	print(sumScore)
	#print('this is resultArr !!!!!!!!!!!!!!!!!!!!!!!!!!!')
	#print(resultArr)
	#resultMat=np.mat(resultArr)        # arr to mat
	#resultMat=resultMat.T           # .T
	resultArr = np.array(resultArr)
	resultList_2=list(resultArr[:,0])+list(resultArr[:,1])+list(resultArr[:,2])+list(resultArr[:,3])+list(resultArr[:,4])+list(resultArr[:,5])+list(resultArr[:,6])
	
	#print('\n')
	print('this result is:    '+fileName+'   new afternoon data prediction!!')
	#print(resultList)
	#print(resultList.shape)
	#print('\n')

	resultDF_2=DataFrame(resultList_2)
	resultDF_2.to_csv(fileName+'_afternoon_predict_0523_v2.csv')

	return sumScore

'''






















'''
gbdt=GradientBoostingRegressor(
  loss='ls'                        #均方误差
, learning_rate=0.05
, n_estimators=100
, subsample=0.6
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
'''










