import sys
import numpy as np
import time
from sklearn.grid_search import ParameterGrid
from sklearn.svm import SVR
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from multiprocessing import Pool, cpu_count, current_process
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from multiprocessing import Pool, cpu_count, current_process
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD
from sklearn.gaussian_process import GaussianProcess
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import copy
from utility.metrics import get_rmse, get_mape, get_mae, get_rae, get_rse, get_mse
from __builtin__ import str

def grid_tuning(params, sdc_data, valid_data, test_data):
    from swsvr.SlidingWindowBasedSVR import SlidingWindowBasedSVR
    errors = Errors()
    test_errors = Errors()
    best_mae = 10000
    if valid_data == None:  sdc_data, valid_data=sdc_data.sep()
    params_grid = list(ParameterGrid(params))
    params_grid = np.random.permutation(params_grid)
    print "+ valid error"
    print "\nIndex, Time, MAE,",
    for k in params_grid[0].keys(): print k+",",
    print ""
    swsvr = SlidingWindowBasedSVR()
    for (index, params) in enumerate(params_grid):
        progress=0
        for i in range(1):
            swsvr.init(svr_cost=params["svr_cost"], svr_epsilon=params["svr_epsilon"],svr_intercept=params["svr_intercept"],svr_itr=params["svr_itr"],
                                kapp_gamma=params["kapp_gamma"], kapp_num=params["kapp_num"],
                                pls_compnum=params["pls_compnum"],
                                sdc_weight=params["sdc_weight"], predict_weight=params["predict_weight"],
                                lower_limit=params["lower_limit"],
                                n_estimators=params["n_estimators"])
            start = time.clock()
            swsvr.train(sdc_data)
            progress = time.clock() - start
            errors.set_errors(valid_data.pre_y, swsvr.predict(valid_data.pre_X), params)
            test_errors.set_errors(test_data.pre_y, swsvr.predict(test_data.pre_X), params)
        if best_mae > errors.tmp_mae:
            best_mae = errors.tmp_mae
            print (str(index)
                   +"/"+str(len(params_grid))
                   +", "+str(progress)
                   +", "+str(errors.tmp_mae)
                   +"("+str(test_errors.tmp_mae)+")"
                   +", "),
            for v in params.values(): print str(v)+",",
            print ""
            sys.stdout.flush()
    errors.print_params()
    #Tune is finished
    best_params = errors.mae_p
    best_swsvr = SlidingWindowBasedSVR(svr_cost=best_params["svr_cost"], svr_epsilon=best_params["svr_epsilon"],svr_intercept=best_params["svr_intercept"],svr_itr=best_params["svr_itr"],
                             kapp_gamma=best_params["kapp_gamma"], kapp_num=best_params["kapp_num"],
                             pls_compnum=params["pls_compnum"],
                             sdc_weight=best_params["sdc_weight"], predict_weight=best_params["predict_weight"],
                             lower_limit=best_params["lower_limit"],
                             n_estimators=params["n_estimators"])
    sdc_data.connect(valid_data)
    best_swsvr.train(sdc_data)
    test_errors = Errors()
    test_errors.set_errors(test_data.pre_y, best_swsvr.predict(test_data.pre_X), errors.mae_p)
    print "+ test error"
    test_errors.print_now()
    return best_swsvr

"""
def grid_tuning_cv(params, sdc_data, cv=10):
    import swsvr.DataStructure as ds

    from swsvr.SlidingWindowBasedSVR import SlidingWindowBasedSVR
    from sklearn.cross_validation import KFold

    params_grid = list(ParameterGrid(params))
    print "\n+ SWSVR CV Tuning:\n"
    print "Number of tuning times: %d"%len(params_grid)
    kf = KFold(len(sdc_data.pre_y), n_folds = cv)
    best_mae = 10000
    best_rmse = 10000
    best_mape = 10000
    print "Index, Time, MAPE(ave), ",
    for k in params_grid[0].keys():
        print k+",",
    print ""
    swsvr = SlidingWindowBasedSVR()
    
    sdc_data.pre_y=np.reshape(sdc_data.pre_y,len(sdc_data.pre_y))
    
    
    for (index, params) in enumerate(params_grid):
        tmp_mae_list=[]
        tmp_rmse_list=[]
        tmp_mape_list=[]
        progress=0
        for train_index, test_index in kf:
            swsvr.init(svr_cost=params["svr_cost"], svr_epsilon=params["svr_epsilon"],svr_intercept=params["svr_intercept"],svr_itr=params["svr_itr"],
                                kapp_gamma=params["kapp_gamma"], kapp_num=params["kapp_num"],
                                pls_compnum=params["pls_compnum"],
                                sdc_weight=params["sdc_weight"], predict_weight=params["predict_weight"],
                                lower_limit=params["lower_limit"],
                                n_estimators=params["n_estimators"])
            start = time.clock()

            #convert train index and test index
            train_fol_X=sdc_data.fol_X[train_index]
            train_pre_X=sdc_data.pre_X[train_index]
            train_pre_y=sdc_data.pre_y[train_index]
            train_sdc_data=ds.SdcData(train_pre_X, train_pre_y, train_fol_X)
            test_fol_X=sdc_data.fol_X[test_index]
            test_pre_X=sdc_data.pre_X[test_index]
            test_pre_y=sdc_data.pre_y[test_index]
            test_sdc_data=ds.SdcData(test_pre_X, test_pre_y, test_fol_X)

            swsvr.train(train_sdc_data) #TODO fix
            progress = time.clock() - start
            swsvr_result = swsvr.predict(test_sdc_data.pre_X)
            tmp_mae, tmp_rmse, tmp_mape = metrics(test_sdc_data.pre_y, swsvr_result)
            tmp_mae_list.append(tmp_mae)
            tmp_rmse_list.append(tmp_rmse)
            tmp_mape_list.append(tmp_mape)
        tmp_mae_list = np.array(tmp_mae_list)
        tmp_rmse_list = np.array(tmp_rmse_list)
        tmp_mape_list = np.array(tmp_mape_list)
        
        if best_mae > np.average(tmp_mae_list):
            print (str(index)
               +", "+str(progress)
               +", "+str(np.average(tmp_mae_list))
               +", "),
            for v in params.values():
                print str(v)+",",
            print ""
            sys.stdout.flush()
            best_params = params
            best_mae = np.average(tmp_mae_list)
            best_rmse = np.average(tmp_rmse_list)
            best_mape = np.average(tmp_mape_list)
    #swsvr._pool_close() #when swsvr run with single process, error appear
    #Tune is finished
    print"\n+ CV detail score:\n"
    print "MAE:"+str(best_mae)+" RMSE:"+str(best_rmse)+" MAPE:"+str(best_mape)+"::"+ str(best_params)
    best_swsvr = SlidingWindowBasedSVR(svr_cost=best_params["svr_cost"], svr_epsilon=best_params["svr_epsilon"],svr_intercept=best_params["svr_intercept"],svr_itr=best_params["svr_itr"],
                             kapp_gamma=best_params["kapp_gamma"], kapp_num=best_params["kapp_num"],
                             pls_compnum=params["pls_compnum"],
                             sdc_weight=best_params["sdc_weight"], predict_weight=best_params["predict_weight"],
                             lower_limit=best_params["lower_limit"],
                             n_estimators=params["n_estimators"])
    best_swsvr.train(sdc_data)
    return best_swsvr
"""

class SimpleLearner:
    
    all_params = {'linearSVR':{'C':[4, 8, 16, 32, 64, 126],'epsilon': [0.001, 0.00001], 'intercept_scaling':  [32, 8, 64, 16, 128]},
                  'svr':{'C':np.logspace(-2, 2, num=5),'epsilon': [0.00001], 'gamma': np.logspace(-6, -4, num=3)},
                  'dt':{'max_depth': np.logspace(1, 5, num=5, base = 2), 'max_features': [x * 0.1 for x in range(1,11) if x%2 == 0]},
                  'knn':{'n_neighbors': [int(x) for x in np.logspace(1, 10, num=10, base=2)] , 'algorithm' : ['auto'], 'leaf_size':[int(x) for x in np.logspace(1, 10, num=10, base=2)]},
                  'ad_svr':{'n_estimators': [int(x) for x in np.logspace(1, 3, num=3)], 'learning_rate': np.logspace(-4, -1, num=4, base = 2), 'loss':['linear','square','exponential']},
                  'ad_dt':{'n_estimators': [int(x) for x in np.logspace(1, 3, num=3)], 'learning_rate': np.logspace(-4, -1, num=4, base = 2), 'loss':['linear','square','exponential']},
                  'bg_svr':{'n_estimators': [int(x) for x in np.logspace(1, 1, num=1)],'max_samples': [x * 0.1 for x in range(1,11) if x%2 == 0],'max_features': [x * 0.1 for x in range(1,11) if x%2 == 0]},
                  'bg_dt':{'n_estimators': [int(x) for x in np.logspace(1, 3, num=3)],'max_samples': [x * 0.1 for x in range(1,11) if x%2 == 0],'max_features': [x * 0.1 for x in range(1,11) if x%2 == 0]},
                  'bg_knn':{'n_estimators': [int(x) for x in np.logspace(1, 3, num=3)],'max_samples': [x * 0.1 for x in range(1,11) if x%2 == 0],'max_features': [x * 0.1 for x in range(1,11) if x%2 == 0]},
                  'gb':{'loss':['huber'],'n_estimators': [10, 100, 300], 'learning_rate': np.logspace(-4, -1, num=4, base = 2),'max_depth': np.logspace(1, 8, num=4, base = 2)},
                  'rf':{'n_estimators': [int(x) for x in np.logspace(1, 5, num=5, base=2)],'max_depth': np.logspace(1, 5, num=5, base = 2),'max_features': [x * 0.1 for x in range(1,11) if x%2 == 0]}
                  }
    """
    #tuned
    all_params = {'linearSVR':{'C':[4],'epsilon': [0.001], 'intercept_scaling':  [128]},
                  'svr':{'C':[100],'epsilon': [0.00001], 'gamma': [0.0001]},
                  'dt':{'max_depth': [32], 'max_features': [1]},
                  'knn':{'n_neighbors': [2] , 'algorithm' : ['auto'], 'leaf_size':[2]},
                  #'ad_svr':{'n_estimators': [int(x) for x in np.logspace(1, 3, num=3)], 'learning_rate': np.logspace(-4, -1, num=4, base = 2), 'loss':['linear','square','exponential']},
                  #'ad_dt':{'n_estimators': [int(x) for x in np.logspace(1, 3, num=3)], 'learning_rate': np.logspace(-4, -1, num=4, base = 2), 'loss':['linear','square','exponential']},
                  #'bg_svr':{'n_estimators': [int(x) for x in np.logspace(1, 1, num=1)],'max_samples': [x * 0.1 for x in range(1,11) if x%2 == 0],'max_features': [x * 0.1 for x in range(1,11) if x%2 == 0]},
                  #'bg_dt':{'n_estimators': [int(x) for x in np.logspace(1, 3, num=3)],'max_samples': [x * 0.1 for x in range(1,11) if x%2 == 0],'max_features': [x * 0.1 for x in range(1,11) if x%2 == 0]},
                  #'bg_knn':{'n_estimators': [int(x) for x in np.logspace(1, 3, num=3)],'max_samples': [x * 0.1 for x in range(1,11) if x%2 == 0],'max_features': [x * 0.1 for x in range(1,11) if x%2 == 0]},
                  'gb':{'loss':['huber'],'n_estimators': [300], 'learning_rate': [0.5],'max_depth': [256]},
                  'rf':{'n_estimators': [32],'max_depth': [32],'max_features': [1]}
                  }
    """
    def __init__(self,name, model):
        '''
        Constructor
        '''
        self.name = name
        self.model = model
        self.result = []
        self.errors = Errors()
        self.test_errors = Errors()

    def predict(self, X):
        self.result = np.array(self.model.predict(X))

    def tuning(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        params_grid = list(ParameterGrid(SimpleLearner.all_params[self.name]))
        print "############",self.name,"############"
        print "+ valid error"
        print "\nIndex, Time, MAE,",
        for k in params_grid[0].keys(): print k+",",
        print ""
        for (index, params) in enumerate(params_grid):
            self.model.set_params(**params)
            start = time.clock()
            self.model.fit(train_x, train_y)
            progress = time.clock() - start
            self.errors.set_errors(valid_y, self.model.predict(valid_x), params)
            # detail
            print (str(index)
                   +"/"+str(len(params_grid))
                   +", "+str(progress)
                   +", "+str(self.errors.tmp_mae)
                   +", "),
            for v in params.values(): print str(v)+",",
            print ""
            sys.stdout.flush()
        self.errors.print_params()
        # train test data and evaluate
        self.model.set_params(**self.errors.mae_p)
        train_x = np.r_[train_x, valid_x]
        train_y = np.r_[train_y, valid_y]
        self.model.fit(train_x, train_y)
        self.test_errors.set_errors(test_y, self.model.predict(test_x), self.errors.mae_p)
        print "+ test error"
        self.test_errors.print_now()
        return self

def other_learners_tuning(sdc_data, valid_data, test_data):
    if test_data == None:
        sdc_data, test_data=sdc_data.sep()
    train_x = sdc_data.pre_X
    train_y = sdc_data.pre_y
    valid_x = valid_data.pre_X
    valid_y = valid_data.pre_y
    test_x = test_data.pre_X
    test_y = test_data.pre_y

    simple_learners = []
    simple_learners.append(SimpleLearner("linearSVR",LinearSVR(dual=False, loss='squared_epsilon_insensitive')).tuning(train_x, train_y,valid_x, valid_y , test_x, test_y))
    simple_learners.append(SimpleLearner("dt",DecisionTreeRegressor()).tuning(train_x, train_y, valid_x, valid_y , test_x, test_y))
    simple_learners.append(SimpleLearner("knn",KNeighborsRegressor()).tuning(train_x, train_y, valid_x, valid_y , test_x, test_y))
    #simple_learners.append(SimpleLearner("ad_dt",AdaBoostRegressor(DecisionTreeRegressor(max_depth=4))).tuning(train_x, train_y, valid_x, valid_y , test_x, test_y))
    #simple_learners.append(SimpleLearner("bg_dt",BaggingRegressor(DecisionTreeRegressor(max_depth=4))).tuning(train_x, train_y, valid_x, valid_y , test_x, test_y))
    #simple_learners.append(SimpleLearner("bg_knn",BaggingRegressor(KNeighborsRegressor())).tuning(train_x, train_y, valid_x, valid_y , test_x, test_y))
    simple_learners.append(SimpleLearner("rf",RandomForestRegressor(n_jobs=-1,)).tuning(train_x, train_y, valid_x, valid_y , test_x, test_y))
    simple_learners.append(SimpleLearner("svr",SVR()).tuning(train_x, train_y, valid_x, valid_y , test_x, test_y))
    simple_learners.append(SimpleLearner("gb",GradientBoostingRegressor()).tuning(train_x, train_y, valid_x, valid_y , test_x, test_y))
    #simple_learners.append(SimpleLearner("ad_svr",AdaBoostRegressor(SVR(kernel='linear',C=1.0, gamma=0, epsilon=0.1))).tuning(train_x, train_y, valid_x, valid_y , test_x, test_y))
    #simple_learners.append(SimpleLearner("bg_svr",BaggingRegressor(SVR(kernel='linear',C=1.0, gamma=0, epsilon=0.1))).tuning(train_x, train_y, valid_x, valid_y , test_x, test_y))

    print "############Tuning Result############"
    for sl in simple_learners:
        print "["+sl.name+"]"+"Best Params:"+str(sl.errors.mae_p)+", Best Error:"+str(sl.errors.mae)
    return simple_learners

class Errors:
    def __init__(self):
        '''
        Constructor
        '''
        self.mae = 9999; self.mae_p = {}; self.tmp_mae = 0; self.mae_list = []
        self.rmse = 9999; self.rmse_p = {}; self.tmp_rmse = 0; self.rmse_list = []
        self.mape = 9999; self.mape_p = {}; self.tmp_mape = 0; self.mape_list = []
        self.mse = 9999; self.mse_p = {}; self.tmp_mse = 0; self.mse_list = []
        self.rse = 9999; self.rse_p = {}; self.tmp_rse = 0; self.rse_list = []
        self.rae = 9999; self.rae_p = {}; self.tmp_rae = 0; self.rae_list = []
    
    def print_now(self):
        str = "(MAE,RMSE,MSE,RSE,RAE,MAPE)=(%f,%f,%f,%f,%f,%f)"%(self.tmp_mae, self.tmp_rmse, self.tmp_mse, self.tmp_rse, self.tmp_rae, self.tmp_mape) 
        print str
        return str
    
    def set_errors(self, y_true, y_pred, params):
        self.tmp_mae = get_mae(y_true, y_pred)
        self.mae_list.append(self.tmp_mae)
        if self.tmp_mae < self.mae: 
            self.mae = self.tmp_mae
            if params != None: self.mae_p = copy.deepcopy(params)
        self.tmp_rmse = get_rmse(y_true, y_pred)
        self.rmse_list.append(self.tmp_rmse)
        if self.tmp_rmse < self.rmse: 
            self.rmse = self.tmp_rmse
            if params != None: self.rmse_p = copy.deepcopy(params)
        self.tmp_mape = get_mape(y_true, y_pred)
        self.mape_list.append(self.tmp_mape)
        if self.tmp_mape < self.mape: 
            self.mape = self.tmp_mape
            if params != None: self.mape_p = copy.deepcopy(params)
        self.tmp_mse = get_mse(y_true, y_pred)
        self.mse_list.append(self.tmp_mse)
        if self.tmp_mse < self.mse: 
            self.mse = self.tmp_mse
            if params != None: self.mse_p = copy.deepcopy(params)
        self.tmp_rse = get_rse(y_true, y_pred)
        self.rse_list.append(self.tmp_rse)
        if self.tmp_rse < self.rse: 
            self.rse = self.tmp_rse
            if params != None: self.rse_p = copy.deepcopy(params)
        self.tmp_rae = get_rae(y_true, y_pred)
        self.rae_list.append(self.tmp_rae)
        if self.tmp_rae < self.rae: 
            self.rae = self.tmp_rae
            if params != None: self.rae_p = copy.deepcopy(params)
        
    def print_params(self):
        print "MAE: %f @ %s" %(self.mae, str(self.mae_p))
        print "RMSE: %f @ %s" %(self.rmse, str(self.rmse_p))
        print "MAPE: %f @ %s" %(self.mape, str(self.mape_p))
        print "MSE: %f @ %s" %(self.mse, str(self.mse_p))
        print "RSE: %f @ %s" %(self.rse, str(self.rse_p))
        print "RAE: %f @ %s" %(self.rae, str(self.rae_p))
        
    def get_last_errors(self):
        return [self.mae_list[-1],
                self.rmse_list[-1],
                self.mape_list[-1],
                self.mse_list[-1],
                self.rse_list[-1],
                self.rae_list[-1],]