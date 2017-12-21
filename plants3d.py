import numpy as np
import cv2
import os
import pandas as pd
from chainer import cuda
from tqdm import tqdm

try:
    cuda.check_cuda_available()
    xp = cuda.cupy
except:
    xp = np

def flick(X):
    return cv2.flip(X, 1)

def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=True,
                              sqrt_bias=0., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).

    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.

    scale : float, optional
        Multiply features by this const.

    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.

    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.

    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.

    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.

    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.

    Notes
    -----
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].

    References
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    X -= mean[:, np.newaxis]  # Makes a copy.
    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X

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

def sep_kaika(dp, area,dateAndTime,ary,sensors,ys, target_num = None):
    pre_x = []
    pre_sensors = []
    pre_y = []
    fol_x = []
    fol_sensors=[]
    fol_y=[]
    fol_dt=[]
    if target_num is None:
        target_kaika_num = int(dp.method.replace("kaika",""))
    else:
        target_kaika_num = target_num
    for ar, dt, a, s, y in zip(area,dateAndTime,ary,sensors,ys):
        if int(dt) <= KAIKA[int(ar)-1][target_kaika_num+1]: # the data until 2016/08/12 12:00 in obj_area is validation data
            #print "[%s] %s -> valid" %(ar, dt)
            pre_x.append(a)
            pre_sensors.append(s)
            pre_y.append(y)
        else: # the data later 2016/08/12 12:00 in obj_area is test data
            #print "[%s] %s -> test" %(ar, dt)
            fol_x.append(a)
            fol_sensors.append(s)
            fol_y.append(y)
            fol_dt.append(str(dt[0:4])+"/"+str(dt[4:6])+"/"+str(dt[6:8])+" "+str(dt[8:10])+":"+str(dt[10:12]))
    return np.array(pre_x), np.array(pre_sensors), np.array(pre_y), np.array(fol_x), np.array(fol_sensors), np.array(fol_y)

class PLANTS():
    def __init__(self, which_set, 
                 path,
                 pname,
                 PATH_TO_DATA_DIR,
                 is_color=False,
                 is_trim=True, 
                 resized_tuple=(50,50), 
                 center=False,
                 axes=['b', 0, 1, 'c'],
                 preprocessor=None,
                 is_gcn=True,
                 fit_preprocessor=False,
                 fit_test_preprocessor=False,
                 is_data_arg=True,
                 is_zca = True,
                 is_rsd = False,
                 target = "pic",
                 start=None,
                 end=None,
                 inArea="all",
                 rate = 0.8,
                 method = "Th",
                 obj_area=[2],
                 thinNum = 1
                 ):
        self.args = locals()
        
        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])
        
        def img2float(paths):
            all_pic = []
            for path in paths:
                try:
                    image = cv2.imread(path)
                except cv2.error:
                    print u'[WARNING] problem happens while processing {}'.format(path)
                    return False
                # coloring
                if is_color==True:
                    processed_img=image
                else:
                    processed_img = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
                all_pic.append(cv2.resize(processed_img, resized_tuple))
            new_pic = np.zeros((resized_tuple[0], resized_tuple[1], 3*len(paths)))
            for i in range(0,len(paths)):
                new_pic[:,:,(i*3):((i*3)+3)] = all_pic[i]
            new_pic = new_pic.astype('uint8')
            new_pic = new_pic.transpose(2,0,1)
            new_pic = new_pic.reshape(new_pic.shape[0]*new_pic.shape[1]*new_pic.shape[2])
            return new_pic
                
        
        def save(img, savepath):
            cv2.imwrite(savepath, img)
            
        def is_in_period(dt):
            if start == 0 and end == 0:
                return True
            if int(dt) > start and int(dt) < end:
                return True
            return False

        ### loading training data ###
        sensors_name=["out_pu", "pu", "pl", "rli", "temp", "humidity"]
        RSD = pd.read_csv(str(path+"/sample.csv"), dtype = {'area':'S10','date':'S10','time':'S10'}, float_precision = "high")

        area = ['0'+str(x) if len(x)==1 else x for x in list(RSD["area"]) ]
        date = list(RSD["date"])
        time = ['0'+str(x) if len(x)==3 else '00'+str(x) if len(x)==2 else '000'+str(x) if len(x)==1 else x for x in list(RSD["time"])]
        check_path = []
        for i in range(0, len(area)):
            check_path.append(str(area[i])+"_"+str(date[i])+"_"+str(time[i])+".jpg")

        sensors_data = []
        for sname in sensors_name:
            sensors_data.append(RSD[str(sname)])
        sensors_data = np.array(sensors_data).T
        if is_rsd: labels = np.array(RSD["rsd"])
        else: labels = np.array(RSD["dsd"])
        dateAndTime = [str(d)+str(t) for d,t in zip(date,time)]
        if len(target) == 1:
            pic1_path = [os.path.join(os.path.abspath(path), target[0], x) for x in check_path]
            pic2_path = pic1_path
            pic3_path = pic1_path
        if len(target) == 2:
            pic1_path = [os.path.join(os.path.abspath(path), target[0], x) for x in check_path]
            pic2_path = [os.path.join(os.path.abspath(path), target[1], x) for x in check_path]
            pic3_path = pic1_path
        if len(target) == 3:
            pic1_path = [os.path.join(os.path.abspath(path), target[0], x) for x in check_path]
            pic2_path = [os.path.join(os.path.abspath(path), target[1], x) for x in check_path]
            pic3_path = [os.path.join(os.path.abspath(path), target[2], x) for x in check_path]

        ### calc maxcount ###
        allDateAndTimeEveryArea = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
        tmp_count_ary=np.zeros(12+1,dtype=np.int)
        for i, (label, pic1p, pic2p, pic3p, dt, sd, ar) in enumerate(zip(labels, pic1_path, pic2_path, pic3_path, dateAndTime, sensors_data, area)):
            if int(dt[10:12]) % thinNum == 0 and os.path.exists(pic1p) and os.path.exists(pic2p) and os.path.exists(pic3p) and is_in_period(dt) and 0 < label and label == label:
                if inArea != "all" and len(np.where(np.array(inArea)==int(ar))[0]) == 0: continue
                tmp_count_ary[int(ar)] += 1
                allDateAndTimeEveryArea[int(ar)].append(dt)                    
        tmp_count_ary=(tmp_count_ary*rate).astype(np.int)
        dt_thresholds=np.zeros(12+1,dtype=np.int)
        for i in range(1,13):
             sorted_dtea = sorted(allDateAndTimeEveryArea[i])
             if len(sorted_dtea) != 0:
                 dt_thresholds[i] = sorted_dtea[tmp_count_ary[i]]
             else:
                 dt_thresholds[i] = 0
                 
        ### following code is added @ 2016/10/20. count data num every days and areas ###
        for i in range(1,13):
            sorted_data = sorted(allDateAndTimeEveryArea[i])
            new_sorted_data=[int(x[:8]) for x in sorted_data]
            sorted_data_uniq =  sorted(list(set(new_sorted_data)))
            print "area%d"%i
            for sdu in range(20160801,20160832):
                leng = len(np.where(np.array(new_sorted_data)==sdu)[0])
                print " +  %d: %d "%(int(sdu),leng)
            pass
        
        # calc max_counts
        this_max_count = 0
        for i, (label, pic1p, pic2p, pic3p, dt, sd, ar) in enumerate(tqdm(zip(labels, pic1_path, pic2_path, pic3_path, dateAndTime, sensors_data, area))):
            if int(dt[10:12]) % thinNum == 0 and os.path.exists(pic1p) and os.path.exists(pic2p) and os.path.exists(pic3p) and is_in_period(dt) and 0 < label and label == label:
                if inArea != "all" and len(np.where(np.array(inArea)==int(ar))[0]) == 0: continue
                if method == "th":
                    if which_set == 'train':
                        if int(dt) > dt_thresholds[int(ar)]:
                            continue
                    elif which_set == 'test':
                        if int(dt) <= dt_thresholds[int(ar)]:
                            continue
                elif "area" in method:
                    if which_set == 'train':
                        if int(ar) in obj_area:
                            continue
                    elif which_set == 'test':
                        if not int(ar) in obj_area:
                            continue
                elif "kaika" in method:
                    if which_set == 'train':
                        if int(dt) > KAIKA[int(ar)-1][int(method.replace("kaika",""))]:
                            continue
                    elif which_set == 'test':
                        if int(dt) <= KAIKA[int(ar)-1][int(method.replace("kaika",""))]:
                            continue
                this_max_count += 1
        print "the number of loading data: %d "%this_max_count
        # load features #
        tmp_labels = []
        topo_view = np.empty((this_max_count, len(target)*3*resized_tuple[0]*resized_tuple[1]), "uint8")
        tmp_dt = []
        tmp_sensors = []
        tmp_area = []
        print_count=0
        for i, (label, pic1p, pic2p, pic3p, dt, sd, ar) in enumerate(tqdm(zip(labels, pic1_path, pic2_path, pic3_path, dateAndTime, sensors_data, area))):
            if int(dt[10:12]) % thinNum == 0 and os.path.exists(pic1p) and os.path.exists(pic2p) and os.path.exists(pic3p) and is_in_period(dt) and 0 < label and label == label:
                if inArea != "all" and len(np.where(np.array(inArea)==int(ar))[0]) == 0: continue
                if method == "th":
                    if which_set == 'train':
                        if int(dt) > dt_thresholds[int(ar)]:
                            continue
                    elif which_set == 'test':
                        if int(dt) <= dt_thresholds[int(ar)]:
                            continue
                elif "area" in method:
                    if which_set == 'train':
                        if int(ar) in obj_area:
                            continue
                    elif which_set == 'test':
                        if not int(ar) in obj_area:
                            continue
                elif "kaika" in method:
                    if which_set == 'train':
                        if int(dt) > KAIKA[int(ar)-1][int(method.replace("kaika",""))]:
                            continue
                    elif which_set == 'test':
                        if int(dt) <= KAIKA[int(ar)-1][int(method.replace("kaika",""))]:
                            continue
                # if differance of labels is needed!!! #
                """
                of_diff_min=10
                if len(np.where(np.array(dateAndTime)==datetime2org(reverse_advance(dt,of_diff_min)))[0]) == 0:
                    continue
                pre_index = np.where(np.array(dateAndTime)==datetime2org(reverse_advance(dt,of_diff_min)))[0][0]
                pre_label = labels[pre_index]
                if pre_label != pre_label: # nan check #
                    continue
                label = label - pre_label
                """
                
                tmp_labels.append(label)
                if len(target) == 1: picp = [pic1p]
                elif len(target) == 2: picp = [pic1p, pic2p]
                elif len(target) == 3: picp = [pic1p, pic2p, pic3p]
                topo_view[print_count] = img2float(picp)
                tmp_dt.append(dt)
                tmp_sensors.append(sd)
                tmp_area.append(ar)
                print_count += 1
                print "["+str(print_count)+"]"+"processing to " + str(pic1p) + " label:" + str(label)
        labels = np.array([[float(x)] for x in tmp_labels])
        dateAndTime = tmp_dt
        sensors_data = np.array(tmp_sensors)
        area = tmp_area
        state = [0] * len(labels)
        """
        #### label normalization ###
        uniq_dateAndTime = np.array(list(set([ x[:8] for x in dateAndTime])))
        uniq_area = np.array(list(set(area)))
        table_max = np.zeros((len(uniq_dateAndTime), len(uniq_area)))
        table_min = np.zeros((len(uniq_dateAndTime), len(uniq_area)))
        tmp_labels = []
        for label, dt, ar in zip(labels, dateAndTime, area):
            dateAndTime_index = np.where(uniq_dateAndTime == dt[:8])[0][0]
            area_index = np.where(uniq_area == ar)[0][0]
            if table_max[dateAndTime_index][area_index] < label[0]:
                table_max[dateAndTime_index][area_index] = label[0]
            if table_min[dateAndTime_index][area_index] > label[0]:
                table_min[dateAndTime_index][area_index] = label[0]
        for label, dt, ar in zip(labels, dateAndTime, area):
            dateAndTime_index = np.where(uniq_dateAndTime == dt[:8])[0][0]
            area_index = np.where(uniq_area == ar)[0][0]
            tmp_labels.append((label[0]-table_min[dateAndTime_index][area_index])/(table_max[dateAndTime_index][area_index]-table_min[dateAndTime_index][area_index]))
            #tmp_labels.append((label[0]-table_min[dateAndTime_index][area_index]))
        labels = np.array([[float(x)] for x in tmp_labels])
        """
        #### Data Augmentation ####
        if is_data_arg:
            # horizontal reflection
            tmp_X = []
            tmp_y = []
            tmp_dt = []
            tmp_state = []
            tmp_sensors = []
            tmp_area = []
            for X,y,dt,sd,ar in zip(topo_view, labels, dateAndTime, sensors_data, area):
                tmp_X.append(flick(X))
                tmp_y.append(y)
                tmp_dt.append(dt)
                tmp_sensors.append(sd)
                tmp_area.append(ar)
                tmp_state.append(1)
            topo_view = np.vstack((topo_view, tmp_X))
            labels = np.vstack((labels, tmp_y))
            dateAndTime.extend(tmp_dt)
            sensors_data = np.vstack((sensors_data, tmp_sensors))
            state.extend(tmp_state)
            area.extend(tmp_area)

        # save image
        if which_set == 'train':
            random_index = np.random.permutation(np.array(range(len(topo_view))))
            for i in range(30):
                if os.path.exists(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, "sample")))==False:os.mkdir(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, "sample")))
                for j in range(len(target)):
                    test_view=topo_view[random_index[i]]#[:,:,(j*3):(j*3+3)]
                    test_view=test_view.reshape(len(target)*3, resized_tuple[0], resized_tuple[1] )
                    test_view=test_view.transpose(1,2,0)[:,:,(j*3):(j*3+3)]
                    save_path = os.path.join(PATH_TO_DATA_DIR, pname, "sample", str(dateAndTime[random_index[i]])+"_"+str(j)+".jpg")
                    save(test_view, save_path)
        
        if topo_view.ndim==3:
            m, r, c = topo_view.shape
            topo_view = topo_view.reshape(m, r, c, 1)
        
        if center:
            topo_view -= topo_view.mean(axis=0)

        ### init of super class ### 
        #self.X = topo_view.transpose(0,3,1,2)
        #self.X = self.X.reshape(topo_view.shape[0],topo_view.shape[1]*topo_view.shape[2]*topo_view.shape[3])
        
        self.X = topo_view
        self.y = labels
        self.dateAndTime = dateAndTime
        self.state = state
        self.sensors_data = sensors_data
        self.area = area

        ### apply gcn ###
        self.X = self.X.astype(xp.float32)
        if is_gcn:
            self.X = global_contrast_normalize(self.X)
        assert not np.any(np.isnan(self.X))
