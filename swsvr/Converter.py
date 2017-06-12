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

class Converter:
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
        #Update info for Converter
        self.features_ave = np.average(sdc_data.pre_X,axis=0)
        self.features_org_std = np.std(sdc_data.pre_X,axis=0)

        #Delete constant features
        self.features_ave = self.delete_constant_feature(self.features_ave, self.features_org_std)
        self.features_std = self.delete_constant_feature(self.features_org_std, self.features_org_std)
        pre_X = self.delete_constant_feature(sdc_data.pre_X, self.features_org_std)
        fol_X = self.delete_constant_feature(sdc_data.fol_X, self.features_org_std)

        #Normalize
        pre_X = self.normalize(pre_X)
        fol_X = self.normalize(fol_X)
        
        if self.is_conv:
            #kernel app
            self.kernel_approximation(pre_X, param_info)
            pre_X = self.apply_kapp(pre_X)
            fol_X = self.apply_kapp(fol_X)
            #PLS
            self.pls(pre_X, sdc_data.pre_y, param_info)
            pls_pre_X = self.apply_pls(pre_X)
            pls_fol_X = self.apply_pls(fol_X)
            self.pls_check_vector = np.std(pls_pre_X,axis=0)
            if len(np.where(self.pls_check_vector == 0)[0]) == 0:
                self.is_pls = True
                pre_X = pls_pre_X
                fol_X = pls_fol_X
            else:
                warnings.filterwarnings("always")
                warnings.warn('Data size must be increased.\n Return SdcData without PLS')
                return ds.SdcData(pre_X, sdc_data.pre_y, fol_X)
        return ds.SdcData(pre_X, sdc_data.pre_y, fol_X)

    def transform_for_features(self, features, time=0):
        """
        Args:
            features: 2dim array
            time: long
        """
        #Delete constant features
        X = self.delete_constant_feature(features, self.features_org_std)

        #Normalize
        X = self.normalize(X)
        
        if self.is_conv:
            #Kapp
            X = self.apply_kapp(X)
            #PLS
            if self.is_pls:
                X = self.apply_pls(X)
        
        if X.shape[0] == 1:
            X = np.reshape(X,len(X[0]))
        return ds.FeaturesData(X, time)

    def transform_for_sdc(self,sdc_data):

        pre_x = self.delete_constant_feature(sdc_data.pre_X, self.features_org_std)
        fol_x = self.delete_constant_feature(sdc_data.fol_X, self.features_org_std)

        #Normalize
        pre_x = self.normalize(pre_x)
        fol_x = self.normalize(fol_x)
        
        if self.is_conv:
            #Kapp
            pre_x = self.apply_kapp(pre_x)
            fol_x = self.apply_kapp(fol_x)
            #PLS
            if self.is_pls:
                pre_x = self.apply_pls(pre_x)
                fol_x = self.apply_pls(fol_x)
        return ds.SdcData(pre_x, sdc_data.pre_y, fol_x)

    def delete_constant_feature(self, X, features_std):
        if X.ndim == 1:
            return np.delete(X, np.where(features_std==0), 0)
        else:
            return np.delete(X, np.where(features_std==0), 1)

    def normalize(self, X):
        X -= self.features_ave
        X /= self.features_std
        return X

    def kernel_approximation(self, x, param_info):
        #kapp = RBFSampler(gamma=param_info.kapp_gamma, param_info.kapp_num)
        kapp = Nystroem(gamma=param_info.kapp_gamma, n_components=param_info.kapp_num)
        self.learned_kapp = kapp.fit(x)

    def apply_kapp(self, x):
        return self.learned_kapp.transform(x)

    def pls(self, x, y, param_info):
        pls = PLSRegression(n_components=param_info.pls_compnum)
        #pls = PLSRegression(n_components=param_info.pls_compnum, max_iter=1000000)
        #pls = PLSSVD(n_components=param_info.pls_compnum)
        self.learned_pls = pls.fit(x,y)

    def apply_pls(self, x):
        return self.learned_pls.transform(x)