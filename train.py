#! /usr/bin/env python
# coding=utf-8

import os
import time
import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import y3.utils as utils
import test
from tqdm import tqdm
from y3.dataset import Dataset
from y3.yolov3 import YOLOv3, decode, compute_loss
from y3.config import cfg
from pandarallel import pandarallel

df_map = pd.DataFrame(columns=['IMAGE',
    'CLASS',
    'SCORE',
    'IOU',
    'BBOXPR',
    'BBOXGT'])

pandarallel.initialize()

def test_step(testing_model, files, df_map):

    CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)
    #classified_image_dir = cfg.TEST.CLASSIFIED_IMAGE_DIR
    #classified_stats_dir = cfg.TEST.CLASSIFIED_STATS_DIR
    #groundtruth_stats_dir = cfg.TEST.GROUNDTRUTH_STATS_DIR

    # run model for metrics collections
    for f in files:
        test_filename = f.split(" ")[0]
        test_bboxes_gt = f.split(" ")[1:]

        predict_result_path = os.path.join(classified_stats_dir, image_name[:-4] + '.txt')


        # delete all entrys of current file from traing_results
        df_map.drop(df_map[df_map.IMAGE == test_filename].index,inplace=True)

        #print(test_bboxes_gt)
        #print('test_filename {}, test_bboxes: {} '.format(test_filename,test_bboxes_gt))
        image = cv2.imread(test_filename)
        image_size = image.shape[:2]
        image_data = utils.image_preporcess(image, [cfg.TEST.INPUT_SIZE, cfg.TEST.INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)


        pred_bbox = testing_model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, cfg.TEST.INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
        # non max supression
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')


        #if cfg.TEST.CLASSIFIED_IMAGE_DIR is not None:
        #    image = utils.draw_bbox(image, bboxes)
        #    cv2.imwrite(classified_image_dir+'/'+test_filename, image)
 
        for bbox in bboxes:
            # create row for every detected bbox:  img, class, score, bbox iou
            test_bbox_gt_class  = [ np.array(x.split(',')[:4], dtype=np.int) for x in test_bboxes_gt if int(x.split(',')[4]) == bbox[5] ]
            test_bbox_gt_class_len = len(test_bbox_gt_class)
            test_bbox_class  = [ x for x in bboxes if x[5] == bbox[5] ]
            # bboxes per image from annotations
            test_bbox_class_len = len(test_bbox_class )
            # compute max iou for current clas
            max_iou_class = max( [ utils.bboxes_iou(bb,bbox[:4]) for bb in test_bbox_gt_class ] or [0] )
        
            df_map = df_map.append({'IMAGE':test_filename, # IMAGE
                    'CLASS':CLASSES[bbox[5]], # CÖASS
                    'SCORE':bbox[4], # SCORE
                    'IOU':max_iou_class, # IOU
                    'BBOXPR':test_bbox_class_len, #BBOXPR
                    'BBOXGT':test_bbox_gt_class_len }, ignore_index=True )    # BBOX

    return df_map        
    
def train_step(image_data, target, model, optimizer, global_steps, warmup_steps, writer, total_steps, epoch):
    
    global df_map

    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss=conf_loss=prob_loss=0
        TP=FP=FN=ALL=0

        # optimizing process
        for i in range(3):

            conv, pred = pred_result[i*2], pred_result[i*2+1]
            # giou_loss, conf_loss, prob_loss,  tp, fp, fn, all
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]
            
        total_loss = giou_loss + conf_loss + prob_loss
   
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> EPOCH:%1.0f STEP %4d   lr: %2.6f   giou_loss: %4.4f   conf_loss: %4.4f   "
            "prob_loss: %4.4f   total_loss: %4.4f " \
                %(epoch, global_steps, optimizer.lr.numpy(),giou_loss, conf_loss, prob_loss, total_loss) )
                  

        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps *cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)

        writer.flush()
        return total_loss


def run(args):
   
    global df_map

    if args.batchsize: cfg.TRAIN.BATCH_SIZE = int(args.batchsize)
    if args.warmupepochs: cfg.TRAIN.WARMUP_EPOCHS = int(args.warmupepochs)
    if args.epochs: cfg.TRAIN.EPOCHS = int(args.epochs)

    if args.gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # create virtual cpu for memory limit
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(args.gpu))]
                )
    else:
        # make gpu invisible
        tf.config.experimental.set_visible_devices([], 'GPU')
   
    num_classes = len(utils.read_class_names(cfg.YOLO.CLASSES))
    trainset = Dataset('train', batch_size=cfg.TRAIN.BATCH_SIZE)
    logdir = cfg.TRAIN.LOG_DIR 
    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch
    #testset=open(cfg.TRAIN.ANNOT_DIR+'test.txt').read().splitlines()


    darknet_input_tensor = tf.keras.layers.Input([
        cfg.TRAIN.INPUT_SIZE[0],cfg.TRAIN.INPUT_SIZE[0],3])

    training_input_tensor = tf.keras.layers.Input([
        cfg.TRAIN.INPUT_SIZE[0],cfg.TRAIN.INPUT_SIZE[0],3])
    #testing_input_tensor = tf.keras.layers.Input([
    #    cfg.TEST.INPUT_SIZE,cfg.TEST.INPUT_SIZE,3])
   
    darknet_conv_tensors = YOLOv3(darknet_input_tensor,80)
    training_conv_tensors = YOLOv3(training_input_tensor,num_classes)
    #testing_conv_tensors = YOLOv3(testing_input_tensor,num_classes)

    # training model
    training_output_tensors = []
    #testing_output_tensors = []
    darknet_output_tensors = []

    for i, b in enumerate(darknet_conv_tensors):
        pt = decode(b, i, 80)
        darknet_output_tensors.append(b)
        darknet_output_tensors.append(pt)

    for i, conv_tensor in enumerate(training_conv_tensors):
        pred_tensor = decode(conv_tensor, i, num_classes)
        training_output_tensors.append(conv_tensor)
        training_output_tensors.append(pred_tensor)
   
    # testing model 
    #for i, fm in enumerate(testing_conv_tensors):
    #    bbox_tensor = decode(fm, i, num_classes)
    #    testing_output_tensors.append(bbox_tensor)


    train_model = tf.keras.Model(training_input_tensor, training_output_tensors)
    #testing_model = tf.keras.Model(testing_input_tensor, testing_output_tensors)

    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    # parse arguments
    if args.tld:
        darknet_model = tf.keras.Model(darknet_input_tensor, darknet_output_tensors)
        utils.load_weights(darknet_model, args.tld)
        
        for i, l in enumerate(darknet_model.layers):
            layer_weights = darknet_model.layers[i].get_weights()
            if layer_weights != []:
                try:
                    train_model.layers[i].set_weights(layer_weights)
                except:
                    print("Skiping",train_model.layers[i].name)
       

        freeze_body=1 # 1 = darknet, 2 = all net excet 3 last layers
        num = (185, len(train_model.layers)-3)[freeze_body-1]
        
        for i in range(num):
           train_model.layers[i].trainable = False
        
        print('Freeze the first {} layers of total {} layers.'.format(num, len(train_model.layers)))

    for epoch in range(cfg.TRAIN.EPOCHS):
        tf.print("=> EPOCH %i OF %i STARTED " % (epoch+1, cfg.TRAIN.EPOCHS))

        all_files = []
        # for batchsize
        for image_data, target, files in trainset:
            total_loss =train_step(image_data, 
                      target, 
                      train_model, 
                      optimizer, 
                      global_steps, 
                      warmup_steps,
                      writer,
                      total_steps,
                      epoch+1
                      )

            all_files.extend(files)

        # save weights after each epoch
        if not math.isnan(total_loss):
            train_model.save_weights(cfg.TRAIN.WEIGHTS_DIR+'Y3')


            # after each epochs 
            if args.map:
                args.gpu = None
                test.run(args,epoch)                

                """
	        # if compute mAP
                testing_model.load_weights(cfg.TRAIN.WEIGHTS_DIR+'Y3')
	
                df_map = test_step(testing_model, testset, df_map) 
                df_map = utils.preprocess_map(df_map, tp_th=0.5)

                if df_map is not None:
                    map_filename = 'ce{}_we{}_e{}_tr{}_te{}_bs{}_test_mAP'.format(
                        epoch+1
                        ,args.warmupepochs
                        ,args.epochs
                        ,len(trainset)
                        ,len(testset)
                        ,args.batchsize
                        )

                    utils.plot_map(df_map,
                        cfg.TRAIN.METRICS_DIR+map_filename+'.png',
                        10,
                        len(utils.read_class_names(cfg.YOLO.CLASSES))
                        )
 
                    df_map.to_csv(cfg.TRAIN.METRICS_DIR+map_filename+'.csv')
                """
            
