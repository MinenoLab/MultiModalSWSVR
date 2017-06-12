'''
Created on 2015/10/05

@author: Kaneda
'''
import itertools
import DataStructure as ds
import Trainer
import NullConverter
import Converter
import ParamInfo
from swsvr import Calculator as ut
import numpy as np
import warnings
import math
from swsvr import DataMaker
import SDC
from multiprocessing import Pool, cpu_count, current_process
from sklearn.cluster import MiniBatchKMeans, MeanShift, KMeans

class SlidingWindowBasedSVR:
    '''Sliding Window based SVR class.

    yklearn specializes to DataMaker series data prediction.
    yklearn builds some specialized models depending on lapse of the DataMaker.
    Specialized model is built by using Short distance Data Collection (SDC).
    Prediction result of yklearn is decided by weighted average.

    Attributes:
    '''

    def __init__(self,
                 svr_cost=1, svr_gamma=0.01, svr_epsilon=0.1, svr_intercept=10, svr_itr=10000, #SVR default
                 kapp_gamma=0.00001, kapp_num=100, #KPCA
                 pls_compnum=10, #pls
                 sdc_weight=0.5, predict_weight=3, #IDW
                 lower_limit=10,
                 n_estimators=10,
                 n_jobs=-1):
        '''yklearn constructor.

        Set some parameters and data for yklearn.
        '''
        if n_jobs <= 0:
            self.p = Pool()
        elif n_jobs > 1:
            self.p = Pool(n_jobs)
        self.weak_learners = []
        self.converters = []
        self.trainer = Trainer.Trainer()
        self.converters.append(NullConverter.NullConverter()) #one converter
        self.param_info = ParamInfo.ParamInfo(svr_cost, svr_gamma, svr_epsilon, svr_intercept, svr_itr,#SVR
                                              kapp_gamma, kapp_num, #KPCA
                                              pls_compnum, #pls
                                              sdc_weight, predict_weight, #IDW
                                              lower_limit,
                                              n_estimators,
                                              n_jobs)
        self.extraction_rate = [] #to evaluate

    def init(self,
                 svr_cost=1, svr_gamma=0.01, svr_epsilon=0.1, svr_intercept=1, svr_itr=1000, #SVR default
                 kapp_gamma=0.00001, kapp_num=100, #KPCA
                 pls_compnum=50, #pls
                 sdc_weight=0.5, predict_weight=3, #IDW
                 lower_limit=10,
                 n_estimators=10,
                 n_jobs=0):
        '''yklearn constructor.

        Set some parameters and data for yklearn.
        '''
        self.weak_learners = []
        self.converters = []
        self.trainer = Trainer.Trainer()
        self.converters.append(NullConverter.NullConverter()) #one converter
        self.param_info = ParamInfo.ParamInfo(svr_cost, svr_gamma, svr_epsilon, svr_intercept, svr_itr,#SVR
                                              kapp_gamma, kapp_num, #KPCA
                                              pls_compnum, #pls
                                              sdc_weight, predict_weight, #IDW
                                              lower_limit,
                                              n_estimators,
                                              n_jobs)
        self.extraction_rate = [] #to evaluate

    def train(self, sdc_data):
        """Set SDC data to yklearn instance.

        Apply SDC data to yklearn. SDC data is defined by DataStructure.py.
        By applying SDC data, yklearn generates new converter so as to convert data.

        Args:
            sdc_data: List of instances of SdcData class.
        """
        self.sdc_data = self.converters[-1].update_and_transform(sdc_data, self.param_info)
        self.sdc_vector = ut.calc_euclid_matrix(self.sdc_data.pre_X, self.sdc_data.fol_X)

        #Kmeans clustering for extracting data to specialize
        #sample_X = np.array(MiniBatchKMeans(verbose=False, compute_labels=False, max_no_improvement=10, n_init=10, init_size=10000, reassignment_ratio=0.001, max_iter=100,  batch_size=5000, n_clusters=int(self.param_info.n_estimators)).fit(self.sdc_data.pre_X).cluster_centers_)
        sample_X = np.array(MiniBatchKMeans(n_clusters=int(self.param_info.n_estimators)).fit(self.sdc_data.pre_X).cluster_centers_)

        #Training
        selfs = [self]*len(sample_X)
        if hasattr(self, "p"):
            result = self.p.map(_parallel_train, zip(selfs, sample_X))
        else:
            result = [self._train(x) for x in sample_X]
            self.extraction_rate = np.array(self.extraction_rate)
            #print "ave:",self.extraction_rate.mean(),", std:",self.extraction_rate.std()
        self.weak_learners.extend(result)

    def _train(self, sam_X):
        dg = ut.calc_euclid_matrix(self.sdc_data.pre_X, [sam_X]*len(self.sdc_data.pre_X))
        training_data = SDC.extract(dg, self.sdc_vector, self.sdc_data, self.param_info)
        self.extraction_rate.append(training_data.y.size*100.0/self.sdc_data.pre_y.size)
        return self.trainer.train(sam_X, training_data, self.param_info, len(self.converters)-1)

    def predict(self, features, varbose=False): #TODO need to optimize
        """Predict Features data using yklearn.

        Predict Features data using weak learners yklearn built. FeaturesData is defined by DataStructure.py.
        Prediction result stlongly depends on the learner built from data similar to features data inpputed.

        Args:
            features: features data(np.array[]).
        """
        pre_value_vector = []
        weight_vector = []
        for i,con in enumerate(self.converters):
            if hasattr(self, "p"):
                tmp_learners = filter(lambda l: l.converter_index == i, self.weak_learners)
                tmp_X = con.transform_for_features(features).X
                tmp_features = [tmp_X]*len(tmp_learners)
                tmp_param_weights = np.array([self.param_info.predict_weight]*len(tmp_learners))
                tmp_pre = self.p.map(_predict, zip(tmp_learners, tmp_features))
                tmp_weight = self.p.map(_calc_euclid, zip(tmp_learners, tmp_features, tmp_param_weights))
            else:
                tmp_features = con.transform_for_features(features).X
                tmp_learners = filter(lambda l: l.converter_index == i, self.weak_learners)
                tmp_pre = map(lambda l: l.predict(tmp_features), tmp_learners)
                tmp_weight = map(lambda l: ut.calc_reciprocal(l.calc_euclid(tmp_features), self.param_info.predict_weight), tmp_learners)
            pre_value_vector.extend(tmp_pre)
            weight_vector.extend(tmp_weight)
        pre_value_vector = np.array(pre_value_vector)
        weight_vector = np.array(weight_vector)
        if varbose:
            return [np.sum(pre_value_vector * weight_vector,axis=0)/np.sum(weight_vector,axis=0), weight_vector, pre_value_vector]
        return np.sum(pre_value_vector * weight_vector,axis=0)/np.sum(weight_vector,axis=0)

    def _pool_close(self):
        self.p.close()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['p']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

def _parallel_train(list):
    me = list[0]
    sam_X = list[1]
    dg = ut.calc_euclid_matrix(me.sdc_data.pre_X, [sam_X]*len(me.sdc_data.pre_X))
    training_data = SDC.extract(dg, me.sdc_vector, me.sdc_data, me.param_info)
    return me.trainer.train(sam_X, training_data, me.param_info, len(me.converters)-1)

def _predict(list):
    l = list[0]
    d = list[1]
    return l.predict(d)

def _calc_euclid(list):
    l = list[0]
    d = list[1]
    w = list[2]
    return ut.calc_reciprocal(l.calc_euclid(d), w)