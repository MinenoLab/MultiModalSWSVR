# coding: utf-8

import datetime
import fnmatch
import gc
import os
import re
import time
import cv2
import numpy as np
import pandas as pd
from numba.decorators import jit
from multiprocessing import Pool
import hasel

imsize = (280, 320)
ksize = 10

# define path
data_root_path = 'dataset'
img_dir_path = os.path.join(data_root_path, 'pic')
mag_dir_path = os.path.join(data_root_path, 'flow_mag') # magnitude of optical flow (csv)
ang_dir_path = os.path.join(data_root_path, 'flow_ang') # angle of optical flow (csv)
out_rgb_flow_path = os.path.join(data_root_path, 'POF')

@jit('f8[:](f8[:],f8[:,:],f8[:])')
def tmp_func(target_max_val, anoter_cnl, target_max_idx):
    anoter_cnl_maxim = np.zeros_like(target_max_val)
    for i in range(anoter_cnl_maxim.shape[0]):
        for j in range(anoter_cnl_maxim.shape[1]):
            anoter_cnl_maxim[i, j] = anoter_cnl[target_max_idx[i, j], i, j]
    return anoter_cnl_maxim

def temporal_maxpooling_core(x):
    """
        args:
         x : 4 dimentions ndarray
        return:
         x_max_pool: 4 dimentions ndarray
    """
    x_mag_max = x[:, :, :, 0]
    x_ang_max = x[:, :, :, 1]

    # calculate max value of magnitude over 5(if ksize=5) image
    mag_max_idx = np.argmax(x_mag_max, axis=0).astype(np.int32)
    mag_max_val = np.max(x_mag_max, axis=0)

    # calculate angle corresponding to max magnitude
    ang_maxim_val = tmp_func(mag_max_val, x_ang_max, mag_max_idx)
    pool_img = np.zeros((imsize[0], imsize[1], 2))
    pool_img[:, :, 0] = mag_max_val
    pool_img[:, :, 1] = ang_maxim_val
    return pool_img


# @jit('f8[:, :, :, :](f8[:, :, :, :], i4)')
def temporal_maxpooling(x, ksize):
    # test(x)[0]
    x_num = x.shape[0] if (x.shape[0] - ksize) > 0 else ksize + 1

    pool_num = x_num - ksize
    ret = np.zeros((pool_num, x.shape[1], x.shape[2], 2))
    for i in range(ksize, x_num):
        # calculate max value over 5(if ksize=5) image each channel
        _x = x[i - ksize: i]
        ret[i - ksize, :, :, :] = temporal_maxpooling_core(_x)

    return ret

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]

def fill_path_core(img_names, is_csv):
    img_names = sorted(img_names)
    time_from = datetime.datetime.strptime(img_names[0][-8:-6] + ':' + img_names[0][-6:-4], '%H:%M')
    time_to = datetime.datetime.strptime(img_names[-1][-8:-6] + ':' + img_names[-1][-6:-4], '%H:%M')
    time_delta = int((time_to - time_from).total_seconds()/60 + 1)

    new_img_names = []
    for i in range(time_delta):
        now = time_from + datetime.timedelta(minutes=i)
        if is_csv:
            new_img_names.append(img_names[0][0:12] + now.strftime('%H%M') + '.csv')
        else:
            new_img_names.append(img_names[0][0:12] + now.strftime('%H%M') + '.jpg')
        # print img_names[0][0:12] + now.strftime('%H%M') + '.jpg'
    new_img_names
    return new_img_names

def fill_path(img_paths, is_csv=True):
    img_names = [os.path.basename(img_path) for img_path in img_paths]

    # extract area and date from image name (ex. 02_20160805_0532.jpg => 02_20160805)
    area_dates = f7([img_name[0:11] for img_name in img_names])

    img_path_fill = []
    for area_date in area_dates:
        # extract images of specific area and date by filtering image names using area_date
        img_names_oneday = fnmatch.filter(img_names, area_date + '*')
        new_img_names = fill_path_core(img_names_oneday, is_csv)
        new_img_paths = [os.path.join(os.path.split(img_paths[0])[0], new_img_name) for new_img_name in new_img_names]
        img_path_fill.extend(new_img_paths)
    return img_path_fill

def extract_path(img_paths, date, area):
    img_dir = os.path.split(img_paths[0])[0]
    img_names = [os.path.basename(img_path) for img_path in img_paths]

    ret = []
    for d in date:
        pattern = area + '_' + d
        repatter = re.compile(pattern)
        ret.append([os.path.join(img_dir, img_name) for img_name in img_names if re.search(repatter, img_name)])
    return ret

def _organize_path(list):
    area = list[0]
    img_names = list[1]
    img_paths = list[2]
    ogn_date = 1
    dates = f7(sorted([img_name[3:11] for img_name in fnmatch.filter(img_names, area + '_*')]))
    organize_paths = []
    print('area {}: {}').format(area, dates)
    for i in range(0, len(dates), ogn_date):
        if i + ogn_date <= len(dates):
            date = dates[i: i + ogn_date]
        else:
            date = dates[i: len(dates) - 1]

        # extract paths corresponding to span that is set at ogn_date
        unit_paths = extract_path(img_paths, date, area)
        organize_paths.append(unit_paths)
    return organize_paths

def pool_organize_path(img_paths, ogn_date=1):
    img_names = [os.path.basename(img_path) for img_path in img_paths]
    areas = f7([img_name[0:2] for img_name in img_names])
    parallel_img_names = [img_names] * len(areas)
    parallel_img_paths = [img_paths] * len(areas)

    p = Pool(10)
    organize_paths = p.map(_organize_path, zip(areas, parallel_img_names, parallel_img_paths))
    p.close()
    p.join()

    new_organize_paths = []
    for one_oroganize_paths in organize_paths:
        new_organize_paths.extend(one_oroganize_paths)

    return new_organize_paths

def _para_load_csv(csv_path):
    if os.path.exists(csv_path):
        csv_src = np.array(pd.read_csv(csv_path, header=None))
    else:
        csv_src = np.zeros((imsize[0], imsize[1]))  # dummy data
    return csv_src

def para_load_csv(csv_paths):
    p = Pool(10)
    ret = p.map(_para_load_csv, csv_paths)
    p.close()
    p.join()
    return np.array(ret)

def rename_ognpath(unit_path):
    return [[os.path.join(img_dir_path, os.path.basename(path)[:-4] + '.jpg') for path in date_path] for date_path in unit_path]

def hsl2rgb_wrap(hsl_mat):
    rgb_list = []
    for i in range(hsl_mat.shape[0]):
        rgb_list.append(hasel.hsl2rgb(hsl_mat[i]))
    return np.array(rgb_list)

#@jit('f8[:](f8[:])')
def ang2ang(img):
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)[0]
    new_img_wide = np.array([x+2*(90-x) if 0 <= x <= 90 else x-2*(x-270) if 270 <= x <= 360 else x for x in img_wide])
    new_img = new_img_wide.reshape(img.shape[0], img.shape[1])
    return new_img

if __name__=='__main__':
    mag_paths = [os.path.join(mag_dir_path, mag_name) for mag_name in fnmatch.filter(os.listdir(mag_dir_path), '*.csv')]
    ang_paths = [os.path.join(ang_dir_path, ang_name) for ang_name in fnmatch.filter(os.listdir(ang_dir_path), '*.csv')]
    mag_paths_fill = fill_path(mag_paths)
    ang_paths_fill = fill_path(ang_paths)
    ogn_mag_paths = pool_organize_path(mag_paths_fill, ogn_date=1)
    ogn_ang_paths = pool_organize_path(ang_paths_fill, ogn_date=1)
    ogn_img_paths = map(rename_ognpath, ogn_mag_paths)

    
    for unit_img_path, unit_mag_path, unit_ang_path in zip(ogn_img_paths, ogn_mag_paths, ogn_ang_paths):
        print('target is {}-{}').format(os.path.basename(unit_mag_path[0][0]), os.path.basename(unit_mag_path[-1][-1]))
        if len(unit_mag_path[0]) < ksize:
            print "this target is too short to pool"
            continue
        for date_img_path, date_mag_path, date_ang_path in zip(unit_img_path, unit_mag_path, unit_ang_path):
            t = time.time()
            flow_mag = para_load_csv(date_mag_path)
            flow_ang = para_load_csv(date_ang_path)

            # convert angle(0-360) to angle(0-180)
            new_flow_ang = np.zeros_like(flow_ang)
            for i in range(flow_ang.shape[0]):
                new_flow_ang[i, :, :] = ang2ang(flow_ang[i, :, :])
            print 'load time: {}'.format(time.time() - t)

            # concatenate optical flow's magnitude and angle
            flow = np.zeros((len(date_mag_path), imsize[0], imsize[1], 2))
            for j in range(flow_mag.shape[0]):
                flow[j, :, :, :] = cv2.merge((flow_mag[j], new_flow_ang[j]))

            # apply max pooling over temporal
            pool_imgs = temporal_maxpooling(flow, ksize)
            pool_img_path = date_img_path[ksize: flow.shape[0]] if ksize < flow.shape[0] else date_img_path[flow.shape[0] - 1:]

            # apply mask to optical flow by filtering mask made from src image
            mask_imgs = pool_imgs

            # make hsl based on pooled image
            mask_imgs[mask_imgs[:, :, :, 0] < 1.5] = 0
            for i in range(mask_imgs.shape[0]): # one image
                mask_imgs[i, :, :, 0] /= np.max(mask_imgs[i, :, :, 0])
            mask_imgs[:, :, :, 1] /= 360

            ##### visualize optical flow #####
            # swap angle and magnitude
            tmp = np.copy(mask_imgs[:, :, :, 0])
            mask_imgs[:, :, :, 0] = mask_imgs[:, :, :, 1]
            mask_imgs[:, :, :, 1] = tmp

            # make hsl and convert that hsl image to rgb image
            l = np.ones((mask_imgs.shape[0], mask_imgs.shape[1], mask_imgs.shape[2], 1)) * 0.5
            hsl_flow = np.c_[mask_imgs, l]
            rgb_flow = hsl2rgb_wrap(hsl_flow)
            ##### visualize optical flow #####

            # save pooled images
            for i, (hf, rf, p) in enumerate(zip(hsl_flow, rgb_flow, pool_img_path)):
                # if image path is not exist, continue
                if np.all(hf[:,:,0] == 0): continue

                rgb_flow_path = os.path.join(out_rgb_flow_path, os.path.basename(p)[:-4] + '.jpg')
                rf = cv2.cvtColor(rf, cv2.COLOR_RGB2BGR)
                cv2.imwrite(rgb_flow_path, rf)

            del flow_mag
            del flow_ang
            del flow
            del pool_imgs
            del mask_imgs
            del hsl_flow
            del rgb_flow
            gc.collect()
