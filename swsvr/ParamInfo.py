'''
Created on 2015/10/05

@author: Kaneda
'''

class ParamInfo:
    '''
    classdocs
    '''

    def __init__(self,
                 svr_cost, svr_gamma, svr_epsilon, svr_intercept, svr_itr,#SVR
                 kapp_gamma, kapp_num, #KPCA
                 pls_compnum,
                 sdc_weight, predict_weight, #IDW
                 lower_limit,
                 n_estimators,
                 n_jobs):
        '''
        Constructor
        '''
        self.svr_cost = svr_cost
        self.svr_gamma = svr_gamma
        self.svr_epsilon = svr_epsilon
        self.svr_intercept = svr_intercept
        self.svr_itr = svr_itr
        self.kapp_gamma = kapp_gamma
        self.kapp_num = kapp_num
        self.pls_compnum = pls_compnum
        self.sdc_weight = sdc_weight
        self.predict_weight = predict_weight
        self.lower_limit = lower_limit
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
