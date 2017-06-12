'''
Created on 2015/10/05

@author: Kaneda
'''
import numpy as np

from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcess

from swsvr.Learner import Learner

class Trainer:
    '''yklearn builds the model by using this class.

    Attributes:
        last_update: unixtime
    '''

    def __init__(self):
        '''Init the Attributes.
        '''
        self.last_update = 0

    def train(self, features, training_data, param_info, converter_index):
        #TODO model_all must be changed deep copy
        """Build the model specialized for current features.

        Args:
            features_data: Instance of FeaturesData class. yklearn specializes this features_data.
            sdc_data: Instance of SdcData class.yklearn builds the model besed on this sdc_data.
            param_info: Instance of ParamInfo class.
            converter_index: int

        Returns:
            Instance of Learner class. This is used as weak learners in yklearn.
        """
        #build model
        #model = GaussianProcess(nugget=10000).fit(training_data.X, training_data.y)
        model = LinearSVR(dual=False, loss='squared_epsilon_insensitive', C=param_info.svr_cost, epsilon=param_info.svr_epsilon, intercept_scaling=param_info.svr_intercept, max_iter=param_info.svr_itr).fit(training_data.X, training_data.y)
        #k = GPy.kern.RBF(input_dim=1)
        #model = GPy.models.GPRegression(training_data.X, training_data.y[:,np.newaxis], k)
        return Learner(features, converter_index, model, training_data.y)

