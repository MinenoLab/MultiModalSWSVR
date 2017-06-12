Name
====
Multi-modal SW-SVR

Overview

This is the implementation for machine learning algorithm "Multi-modal SW-SVR"

## Requirement

chainer: https://github.com/chainer/

## Usage

step.1: download sample dataset from following URL and set it to root path
XXX


step.2: make pooled optical flow from optical flow
python temporal_pooling.py

step.3: make ROAF by applying pooled optical flow to original image using for calculating optical flow
python apply_mask_for_orgimg.py

step.4: train Multi-modal SW-SVR
python train.py

## References
Kaneda, Y., & Mineno, H. (2016). Sliding window-based support vector regression for predicting micrometeorological data.
Expert System with Application, 59, 217?225. http://doi.org/10.1016/j.eswa.2016.04.012
