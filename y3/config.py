#! /usr/bin/env python
# coding=utf-8

import os
from easydict import EasyDict as edict
from datetime import datetime

ENV_DATA='Y3_DATA'

if ENV_DATA in os.environ :

    path = os.environ[ENV_DATA]
    
    # read, split and strip lines from file
    conf = [ n.strip() for i in open(path+'/conf/config.py').readlines()
        if not i.startswith('#') and not i.startswith('\n') for n in i.split('=') ]
    conf = dict(zip(conf[::2], conf[1::2] ))
else:
    print('ERROR: ENV Variable {} does not exist.'.format(ENV_DATA))
    exit(1)

path_last_run  = path+'/'+datetime.now().strftime("%d%m%Y_%H%M")

__C                           = edict()
# Consumers can get config by: from config import cfg
cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

# Set the class name
__C.YOLO.DIR                  = path+'/'
__C.YOLO.WEIGHTS_DIR          = conf['CONFIG_WEIGHTS_DIR']+'/'
__C.YOLO.WEIGHTS_LOADED       = conf['CONFIG_WEIGHTS_DIR']+'/'+conf['CONFIG_WEIGHTS_LOADED']
__C.YOLO.WEIGHTS_COMPUTED     = conf['CONFIG_WEIGHTS_DIR']+'/'+conf['CONFIG_WEIGHTS_COMPUTED']

__C.YOLO.LOG                  = conf['CONFIG_LOGDIR']
__C.YOLO.CLASSES              = path+conf['CONFIG_CLASSES']
__C.YOLO.ANCHORS              = path+conf['CONFIG_ANCHORS']
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = path+conf['TRAIN_ANNOT_PATH']
__C.TRAIN.BATCH_SIZE          = int(conf['TRAIN_BATCH_SIZE'])
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = [int(conf['TRAIN_INPUT_SIZE'].strip(']['))]
__C.TRAIN.DATA_AUG            = bool(conf['TRAIN_DATA_AUG'])
__C.TRAIN.LR_INIT             = float(conf['TRAIN_LR_INIT'])
__C.TRAIN.LR_END              = float(conf['TRAIN_LR_END'])
__C.TRAIN.WARMUP_EPOCHS       = int(conf['TRAIN_WARMUP_EPOCHS'])
__C.TRAIN.EPOCHS              = int(conf['TRAIN_EPOCHS'])



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = path+conf['TEST_ANNOT_PATH']
__C.TEST.BATCH_SIZE           = int(conf['TEST_BATCH_SIZE'])
__C.TEST.INPUT_SIZE           = int(conf['TEST_INPUT_SIZE'])
__C.TEST.DATA_AUG             = bool(conf['TEST_DATA_AUG'])
__C.TEST.CLASSIFIED_IMAGE_DIR = conf['TEST_CLASSIFIED_IMAGE_DIR']
__C.TEST.CLASSIFIED_STATS     = conf['TEST_CLASSIFIED_STATS']
__C.TEST.GROUNDTRUTH_STATS    = conf['TEST_GROUNDTRUTH_STATS']
__C.TEST.SCORE_THRESHOLD      = float(conf['TEST_SCORE_THRESHOLD'])
__C.TEST.IOU_THRESHOLD        = float(conf['TEST_IOU_THRESHOLD'])

