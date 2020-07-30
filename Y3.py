#! /usr/bin/env python
# coding=utf-8

import os
import argparse
#import pathlib
#from datetime import datetime
#from shutil import copyfile

ENV_TRAIN  = 'Y3_TRAIN'

parser = argparse.ArgumentParser(add_help=False)
#parser.add_argument("--train",help="Train this model", default=False, action='store_true')
parser.add_argument("--test",help="Test this model", default=False, action='store_true')
parser.add_argument("--trainpath",help="Absolute path to Y3 data dir")
#parser.add_argument("--train_dir",help="subdirectory for storing weights and configs for certain training run")
parser.add_argument("--tld", help="Use transfer learing for darknet, train the rest")
parser.add_argument("--gpu", help="Use local GPU with certain max amount of GPU memory in MB",nargs='?',const=1500)
parser.add_argument("--map", help="compute mAP and save as image", default=False, action='store_true')
parser.add_argument("--batch_size", help="Which batch to use in training, default=10",nargs='?',const=10)

args = parser.parse_args()

if args.trainpath  or (args.test and args.test and args.train_dir):

    if os.path.exists(args.trainpath):
        args.trainpath = args.trainpath+'/' if not args.trainpath.endswith('/') else args.trainpath 

        os.environ[ENV_TRAIN] = args.trainpath

        # create dedicated directory
        training_subdir = ['log','weights','stats','metrics']
        if not os.path.exists(args.trainpath):
            os.mkdir(args.trainpath)

        for x in training_subdir:
            if not os.path.exists(args.trainpath+'/'+x):
                os.mkdir(args.trainpath+'/'+x) 

        if args.trainpath:
            import train 
            train.run(args)

        if args.test:
            if not os.path.exists(args.path):
                print('Directory {} does not exist'.format(args.path))
                exit(1)

            import test 
            test.run(args)

    else:
        print("File does not exist"+args.trainpath+'/conf/config.py')
        exit()

else:
    parser.print_help()
