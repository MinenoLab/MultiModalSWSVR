'''
Created on 2015/10/05

@author: Kaneda
'''
from numba.decorators import jit
from swsvr import Calculator as ut

class Learner:
    '''
    Weak learner in yklearn. Some model in scikit-learn can be applied.
    e.g. SVR, Decision Tree, kNN.

    Attributes:
        specialized_features: features data(np.array[])
        converter_index: int
        model: model in scikit-learn
    '''

    def __init__(self, specialized_features, converter_index, model, train_y):
        '''Init the attributes.

        Args:
            specialized_features: features data(np.array[])
            converter_index: int
            model: model in scikit-learn
        '''
        self.specialized_features = specialized_features
        self.converter_index = converter_index
        self.model = model
        self.train_y = train_y

    def predict(self, features):
        '''Predict value using self.model.

        Args:
            features: features data(np.array[])
        '''
        return self.model.predict(features)

    def calc_euclid(self, features):
        '''calculate eunclid distance between features and self.specialized_features

        Args:
            features: features data(np.array[])
        '''
        tmp = [self.specialized_features]*len(features)
        return ut.calc_euclid_matrix(features, tmp)