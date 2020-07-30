#! /usr/bin/env python
# coding=utf-8

import os
import pathlib
from easydict import EasyDict as edict


ENV_TRAIN='Y3_TRAIN'

if ENV_TRAIN in os.environ :
    trainpath  = os.environ[ENV_TRAIN]
    scriptpath = str(pathlib.Path(__file__).parent.absolute())+'/'

else:
    print('ERROR: ENV Variable {} does not exist.'.format(ENV_TRAIN))
    exit(1)


__C                           = edict()
cfg                           = __C
__C.YOLO                      = edict()
__C.TRAIN                     = edict()
__C.TEST                      = edict()
__C.TEST.DIR                  = trainpath
__C.TRAIN.DIR                 = trainpath
__C.YOLO.SCRIPTDIR            = scriptpath

# Set the class name
__C.YOLO.CLASSES              = trainpath+"conf/classes.names"
__C.YOLO.ANCHORS              = scriptpath+"anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.3

# Train options
__C.TRAIN.LOG_DIR             = __C.TRAIN.DIR+"log/"
__C.TRAIN.WEIGHTS_DIR         = __C.TRAIN.DIR+"weights/"
__C.TRAIN.METRICS_DIR         = __C.TRAIN.DIR+'metrics/'
__C.TRAIN.ANNOT_DIR           = __C.TRAIN.DIR+"conf/"
__C.TRAIN.BATCH_SIZE          = 10 
#TRAIN_INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = [416]
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-4
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 1 
__C.TRAIN.EPOCHS              = 10 

# TEST options
__C.TEST.ANNOT_PATH           = __C.TEST.DIR+"conf/test.txt"
__C.TEST.BATCH_SIZE           = 1 
__C.TEST.INPUT_SIZE           = 416 
__C.TEST.DATA_AUG             = False
__C.TEST.CLASSIFIED_IMAGE_DIR = __C.TEST.DIR+"images/classified"
__C.TEST.GROUNDTRUTH_IMAGE_DIR= __C.TEST.DIR+"images/groundtruth"
__C.TEST.CLASSIFIED_STATS     = __C.TEST.DIR+"stats/classified"
__C.TEST.GROUNDTRUTH_STATS    = __C.TEST.DIR+"stats/groundtruth"
__C.TEST.SCORE_THRESHOLD      = 0.30
__C.TEST.IOU_THRESHOLD        = 0.30
