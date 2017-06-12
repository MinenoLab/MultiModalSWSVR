'''
Created on 2015/10/05

@author: Kaneda
'''
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD
from blaze.compute.numpy import axify
import DataStructure as ds
from swsvr import Calculator as ut
import warnings
from sklearn.cluster import MiniBatchKMeans, MeanShift
from sklearn.kernel_approximation import RBFSampler, Nystroem

class NullConverter:
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.is_kpca = False
        self.is_pls = False
        
        #debug param
        self.is_conv = True

    def update_and_transform(self, sdc_data, param_info):
        return ds.SdcData(sdc_data.pre_X, sdc_data.pre_y, sdc_data.fol_X)

    def transform_for_features(self, features, time=0):
        return ds.FeaturesData(features, time)

    def transform_for_sdc(self,sdc_data):
        return ds.SdcData(sdc_data.pre_X, sdc_data.pre_y, sdc_data.fol_X)
