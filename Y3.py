#! /usr/bin/env python
# coding=utf-8

import os
import argparse
from datetime import datetime

ENV_DATA = 'Y3_DATA'
ENV_LR   = 'Y3_LASTRUN'

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--train",help="Train this model", default=False, action='store_true')
parser.add_argument("--test",help="Test this model", default=False, action='store_true')
parser.add_argument("--path",help="Absolute path to Y3 data dir")
parser.add_argument("--train_dir",help="subdirectory for storing weights and configs for certain training run")
parser.add_argument("--tld", help="Use transfer learing for darknet, train the rest", default=False, action='store_true')
parser.add_argument("--gpu", help="Use local GPU with certain max amount of GPU memory in MB",nargs='?',const=1500)
parser.add_argument("--map", help="compute mAP and save as image", default=False, action='store_true')


args = parser.parse_args()

if args.path and (args.train  or (args.test and args.test and args.train_dir)):

    if os.path.exists(args.path+'/conf/config.py'):
        #print(args.path)
        args.path = args.path+'/' if not args.path.endswith('/') else args.path 
        now = datetime.now().strftime("%d%m%Y_%H%M")

        os.environ[ENV_DATA] = args.path
        os.environ[ENV_LR] = args.path+now

        # if we train and train_dir was not specified, take now direcotry
        if not args.train_dir:
            args.train_dir = now

        # create dedicated directory
        training_subdir = ['log','weights','stats','metrics']
        if not os.path.exists(args.path+args.train_dir):
            os.mkdir(args.path+args.train_dir)

        for x in training_subdir:
            if not os.path.exists(args.path+args.train_dir+'/'+x):
                os.mkdir(args.path+args.train_dir+'/'+x) 

        if args.train:
            import train 
            train.run(args)

        if args.test:
            if not os.path.exists(args.path+args.train_dir):
                print('Directory {} does not exist'.format(args.path+args.train_dir))
                exit(1)

            import test 
            test.run(args)

    else:
        print("File does not exist"+args.path+'/conf/config.py')
        exit()

else:
    parser.print_help()
