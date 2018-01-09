# Multi-modal SW-SVR

## Overview

This is the implementation for machine learning algorithm "Multi-modal SW-SVR"

## Requirement
Python 2.7.12 :: Anaconda custom (64-bit)  
chainer ver.1.19.0  
Scikit-learn ver.0.18.1  

## Usage

step.1: download sample dataset and rename the directory name to "dataset"
<http://www.minelab.jp/public_data/plant_wilt_sample.tar.gz>

step.2: make pooled optical flow from optical flow

```
$ python temporal_pooling.py
```

step.3: make ROAF by applying pooled optical flow to original image using for calculating optical flow

```
$ python apply_mask_for_orgimg.py
```

step.4: train Multi-modal SW-SVR

```
$ python train.py
```

## Demo
![Demo](https://user-images.githubusercontent.com/10162931/34709936-b08fad72-f55c-11e7-828e-7f64210cd1dc.png)


## References
[1] Kaneda, Y., Shibata S. & Mineno, H.: Multi-modal sliding window-based support vector regression for predicting plant water stress, Knowledge-based Systems (KNOSYS), pp. 135-148, https://doi.org/10.1016/j.knosys.2017.07.028  
[2] Kaneda, Y. & Mineno, H.: Sliding window-based support vector regression for predicting micrometeorological data. Expert System with Application, 59, 217?225. http://doi.org/10.1016/j.eswa.2016.04.012 
