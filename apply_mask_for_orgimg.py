'''
Created on Dec 10, 2016

@author: minelab
'''
import os
import numpy as np
import cv2
from numba.decorators import jit

@jit('u1[:,:,:](u1[:,:,:],u1[:,:,:])')
def apply_mask(rgb_img, pool_img):
    s_img = np.dstack([pool_img[:,:,2],pool_img[:,:,2],pool_img[:,:,2]])
    s_img[s_img<=50] = 0; s_img[s_img>50] = 1 
    rgb_img = rgb_img*s_img
    return rgb_img

def apply_mask_kido(rgb_img, pool_img):
    rbg_img = cv2.cvtColor(rgb_img.astype('uint8').copy(), cv2.COLOR_BGR2HLS)
    rgb_img[:,:,1] = pool_img[:,:,2]*0.8
    return cv2.cvtColor(rgb_img, cv2.COLOR_HLS2RGB)

def getFileList(orgpath, ppath, outpath):
    for (root, dirs, files) in os.walk(orgpath):
        for file in files:
            if os.path.exists(os.path.join(ppath, file)):
                print os.path.join(ppath,file)
                orgimage = cv2.imread(os.path.join(orgpath,file))
                pimage = cv2.imread(os.path.join(ppath,file))
                orgimage=cv2.resize(orgimage,(224,224))
                pimage=cv2.resize(pimage,(224,224))
                maskedimg = apply_mask(orgimage.astype('uint8'),
                                            cv2.cvtColor(pimage.astype('uint8').copy(), cv2.COLOR_BGR2HLS))
                cv2.imwrite(os.path.join(outpath,file), maskedimg)

if __name__ == '__main__':
    rootpath = "dataset"
    orgpath = os.path.join(rootpath, "pic")
    ppath = os.path.join(rootpath, "POF")
    outpath = os.path.join(rootpath, "ROAF")
    getFileList(orgpath, ppath, outpath)
    
     
