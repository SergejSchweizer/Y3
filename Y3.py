#! /usr/bin/env python
# coding=utf-8

import os
import argparse
#import pathlib
#from datetime import datetime
#from shutil import copyfile

ENV_Y3_DIR  = 'Y3_DIR'

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--testpath",help="Absolut path to Y3 data dir")
parser.add_argument("--trainpath",help="Absolute path to Y3 data dir")
parser.add_argument("--tld", help="Use transfer learing for darknet, train the rest")
parser.add_argument("--gpu", help="Use local GPU with certain max amount of GPU memory in MB",nargs='?',const=1500)
parser.add_argument("--map", help="compute mAP and save as image", default=False, action='store_true')
parser.add_argument("--batchsize", help="Which batch to use in training, default=10",nargs='?',const=10)
parser.add_argument("--warmupepochs", help="Increasing learnig rate at the beginning",nargs='?',const=2)
parser.add_argument("--epochs", help="How many epcohs should we train",nargs='?',const=10)


args = parser.parse_args()

if args.trainpath:

    if os.path.exists(args.trainpath):
        args.trainpath = args.trainpath+'/' if not args.trainpath.endswith('/') else args.trainpath 

        os.environ[ENV_Y3_DIR] = args.trainpath

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

elif args.testpath :
    os.environ[ENV_Y3_DIR] = args.testpath

    if os.path.exists(args.testpath):
        args.testpath = args.testpath+'/' if not args.testpath.endswith('/') else args.testpath 
        
        import test 
        test.run(args)

else:
    print("File does not exist"+args.trainpath+'/conf/config.py')
    parser.print_help()
