
import os
import time
import warnings
from time import strftime
from functools import partial

from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from skimage import io as skio
from scipy.special import expit, softmax

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as func

from datasets import FetalDataset
from models import FCN_ResNet50, FCN_ResNet101, LRASPP_MobileNet
from models import DeeplabV3_MobileNet, DeeplabV3_ResNet50, Unet2D
from utils import color_codes, time_to_string, normalise

from pathlib import Path

from datasets import FetalDataset
from experiments  import run_segmentation_experiments_fetal

import sys


if __name__ == "__main__":

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    master_seed = 42

    train_batch = 8
    test_batch = 8
    epochs = 50
    patience = 50
    verbP = 1
    name = 'fetal us'
    classes = ['background', "third_sylvian", "third_midline", "third_cavum", "third_cerebellum"]

    d_path = "./data"
    path_training = sys.argv[1]
    path_testing = sys.argv[2]
 
    gimpFormat = False # to see if the images were in gimp format or had already been separated

    lDict = {"third_sylvian" : (1,False), "third_midline": (2,False), "third_cavum": (3,True), "third_cerebellum":(4,True)} 
    #lDict = {"third_sylvian" : (1,False), "third_midline": (2,False), "third_cavum": (3,False), "third_cerebellum":(4,False)}  # for only line segmentation

    training_set = FetalDataset(path_training, gimpFormat = gimpFormat, lDict = lDict )
    print("training data set made ")
    testing_set = FetalDataset(path_testing, gimpFormat = gimpFormat, lDict = lDict)
    print("testing data set made "+str(len(testing_set)))
    # The experiments are run next. We capture some warnings related to
    # image loading to clean the debugging console.
        
    print("training unet")    

    # Unet [64, 64, 256, 256, 512, 512]
    unet_dsc, unet_k_dsc = run_segmentation_experiments_fetal(
        master_seed, 'unet2d', 'Unet 2D', name,
        partial(Unet2D, lr=1e-4, conv_filters=[64, 64, 256, 256, 512, 512]),
        training_set, testing_set, 
        os.path.join(d_path, 'Weights'), os.path.join(d_path, 'Predictions'),
        classes, n_inputs = 3, n_classes = 5, epochs=epochs, patience=patience,
        train_batch=train_batch, test_batch=test_batch, verbose = verbP
    )

    print("FINISEHD UNET TRAINING, RESULTS: ")
    print(unet_dsc)
    print(unet_k_dsc)
    print("************************************************************************************* ")


    # FCN ResNet50
    fcn50_dsc, fcn50_k_dsc = run_segmentation_experiments_fetal(
        master_seed, 'fcn-resnet50', 'FCN ResNet50', name,
        partial(FCN_ResNet50, lr=1e-4, pretrained=True),
        training_set, testing_set,
        os.path.join(d_path, 'Weights'), os.path.join(d_path, 'Predictions'),
        classes, n_inputs = 3, n_classes = 5, epochs=epochs, patience=patience,
        train_batch=train_batch, test_batch=test_batch, verbose = verbP
    )

    print("FINISEHD FCN RESNET50 TRAINING, RESULTS: ")
    print(fcn50_dsc)
    print(fcn50_k_dsc)
    print("************************************************************************************* ")

    train_batch = 4
    test_batch = 4

    # FCN ResNet101
    fcn101_dsc, fcn101_k_dsc = run_segmentation_experiments_fetal(
        master_seed, 'fcn-resnet101', 'FCN ResNet101', name,
        partial(FCN_ResNet101, lr=1e-4, pretrained=True),
        training_set, testing_set,
        os.path.join(d_path, 'Weights'), os.path.join(d_path, 'Predictions'),
        classes, n_inputs = 3, n_classes = 5, epochs=epochs, patience=patience,
        train_batch=train_batch, test_batch=test_batch, verbose = verbP
    )

    print("FINISEHD FCN RESNET50 TRAINING, RESULTS: ")
    print(fcn101_dsc)
    print(fcn101_k_dsc)
    print("************************************************************************************* ")


    # DeeplapV3 MobileNet
    dl3mn_dsc, dl3mn_k_dsc = run_segmentation_experiments_fetal(
        master_seed, 'deeplab3-mobilenet', 'DeeplabV3 MobileNet', name,
        partial(DeeplabV3_MobileNet, lr=1e-4, pretrained=True),
        training_set, testing_set,
        os.path.join(d_path, 'Weights'), os.path.join(d_path, 'Predictions'),
        classes, n_inputs = 3, n_classes = 5, epochs=epochs, patience=patience,
        train_batch=train_batch, test_batch=test_batch, verbose = verbP
    )

    print("FINISEHD DLMOB TRAINING, RESULTS: ")
    print(dl3mn_dsc)
    print(dl3mn_dsc)
    print("************************************************************************************* ")


    # DeeplapV3 ResNet50
    dl3rn_dsc, dl3rn_k_dsc = run_segmentation_experiments_fetal(
        master_seed, 'deeplab3-resnet50', 'DeeplabV3 ResNet50', name,
        partial(DeeplabV3_ResNet50, lr=1e-4, pretrained=True),
        training_set, testing_set,
        os.path.join(d_path, 'Weights'), os.path.join(d_path, 'Predictions'),
        classes, n_inputs = 3, n_classes = 5, epochs=epochs, patience=patience,
        train_batch=train_batch, test_batch=test_batch, verbose = verbP
    )

    print("FINISEHD DL50 TRAINING, RESULTS: ")
    print(dl3rn_dsc)
    print(dl3rn_dsc)
    print("************************************************************************************* ")


    # L-RASPP ResNet50
    lraspp_dsc, lraspp_k_dsc = run_segmentation_experiments_fetal(
        master_seed, 'lraspp-mobilenet', 'Lite R-ASPP MobileNet', name,
        partial(LRASPP_MobileNet, lr=1e-4, pretrained=True),
        training_set, testing_set,
        os.path.join(d_path, 'Weights'), os.path.join(d_path, 'Predictions'),
        classes, n_inputs = 3, n_classes = 5, epochs=epochs, patience=patience,
        train_batch=train_batch, test_batch=test_batch, verbose = verbP
    )


    print("FINISEHD Lrasp TRAINING, RESULTS: ")
    print(lraspp_dsc)
    print(lraspp_k_dsc)
    print("************************************************************************************* ")
