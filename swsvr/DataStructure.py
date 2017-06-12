from astropy.units import second
import numpy as np

KAIKA = [[201608081200,201608121200,201608151200,201608241200,201608291200,201609051200],#1
         [201608081200,201608121200,201608191200,201608241200,201609021200,201609091200],#2
         [201608081200,201608121200,201608151200,201608241200,201608291200,201609051200],#3
         [201608081200,201608121200,201608171200,201608241200,201608291200,201609051200],#4
         [201608081200,201608121200,201608151200,201608171200,201608191200,201608261200],#5
         [201608081200,201608121200,201608171200,201608191200,222222221200,222222221200],#6
         [201608081200,201608121200,201608171200,201608221200,201608291200,201609051200],#7
         [201608081200,201608121200,201608101200,201608151200,201608241200,201608311200],#8
         [201608081200,201608121200,201608121200,201608191200,201608261200,201609021200],#9
         [201608081200,201608121200,201608171200,201608191200,201608241200,201608311200],#10
         [201608081200,201608121200,201608121200,201608191200,201608221200,201608291200],#11
         [201608081200,201608121200,201608101200,201608151200,201608221200,201608291200]]#12

class FeaturesData:
    '''
    This is features set. This is implemented by numpy.
    '''

    def __init__(self, X, time=[]):
        '''
        Constructor
        '''
        self.X = X #2dim array
        self.time = time #1dim array

class LabeledData:
    '''
    This is training data for supervised learning.
    This is FeaturesData with label.
    '''

    def __init__(self, X, y, time=[]):
        '''
        Constructor
        '''
        self.X = X #2dim array
        self.y = y #1dim array
        self.time = time #1dim array

class SdcData:
    '''
    This is data for SDC. SDC needs two LabeledData, previous data and following data.
    e.g. in 6 hours later prediction, the SDC data has 13:00 data and 19:00 data.
    '''

    def __init__(self, pre_X, pre_y, fol_X, pre_XS=None, fol_XS=None):
        '''
        Constructor
        '''
        # main variable
        self.pre_X = pre_X #2dim array
        self.pre_y = pre_y #1dim array
        self.fol_X = fol_X #2dim array
        
        # dummy variable for sensor data
        self.pre_XS = pre_XS #2dim array
        self.fol_XS = fol_XS #2dim array
        
    def sep(self, rate=0.8):
        index=len(self.pre_X)*rate
        if self.pre_XS is not None:
            farst_SDC = SdcData(self.pre_X[:index], np.reshape(self.pre_y,len(self.pre_y))[:index], self.fol_X[:index], self.pre_XS[:index],self.fol_XS[:index])
            second_SDC = SdcData(self.pre_X[index:], np.reshape(self.pre_y,len(self.pre_y))[index:], self.fol_X[index:], self.pre_XS[index:],self.fol_XS[index:])        
            return farst_SDC, second_SDC
        else:
            farst_SDC = SdcData(self.pre_X[:index], np.reshape(self.pre_y,len(self.pre_y))[:index], self.fol_X[:index])
            second_SDC = SdcData(self.pre_X[index:], np.reshape(self.pre_y,len(self.pre_y))[index:], self.fol_X[index:])        
            return farst_SDC, second_SDC
    
    def connect(self, sdcd):
        self.pre_X = np.r_[self.pre_X, sdcd.pre_X]
        self.pre_y = np.r_[self.pre_y, sdcd.pre_y]
        self.fol_X = np.r_[self.fol_X, sdcd.fol_X]
    
    def sep_kaika_sdc(self, dp, area, dateAndTime, target_num = None):
        first_pre_x = []
        first_pre_y = []
        first_fol_x = []
        
        second_pre_x = []
        second_pre_y = []
        second_fol_x = []
        
        if target_num is None:
            target_kaika_num = int(dp.method.replace("kaika",""))
        else:
            target_kaika_num = target_num
        for ar, dt, pre_X, pre_y, fol_X in zip(area,dateAndTime, self.pre_X, self.pre_y, self.fol_X):
            if int(dt) <= KAIKA[int(ar)-1][target_kaika_num+1]:
                print "[%s] %s -> valid" %(ar, dt)
                first_pre_x.append(pre_X)
                first_pre_y.append(pre_y)
                first_fol_x.append(fol_X)
            else:
                print "[%s] %s -> test" %(ar, dt)
                second_pre_x.append(pre_X)
                second_pre_y.append(pre_y)
                second_fol_x.append(fol_X)
        first_SDC = SdcData(np.array(first_pre_x), np.array(first_pre_y), np.array(first_fol_x))
        second_SDC = SdcData(np.array(second_pre_x), np.array(second_pre_y), np.array(second_fol_x))
        return first_SDC, second_SDC
