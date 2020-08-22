#! /usr/bin/env python
# coding=utf-8

import cv2
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import y3.utils as utils
from y3.config import cfg
from y3.yolov3 import YOLOv3, decode


df_map = pd.DataFrame(columns=['IMAGE',
    'CLASS',
    'SCORE',
    'IOU',
    'BBOXPR',
    'BBOXGT'])



def run(args,epoch,prefix_filename):
   
    global df_map

   
    if args.gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # create virtual cpu for memory limit
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(args.gpu))]
                )
    #else:
    #    # make gpu invisible
    #    tf.config.experimental.set_visible_devices([], 'GPU')
   


    INPUT_SIZE   = cfg.TEST.INPUT_SIZE 
    NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
    CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)
    TRAIN_SIZE   = len(open(cfg.TRAIN.ANNOT_DIR+'train.txt').readlines())
    trainset=open(cfg.TRAIN.ANNOT_DIR+'train.txt').read().splitlines()
    testset=open(cfg.TRAIN.ANNOT_DIR+'test.txt').read().splitlines()

    classified_stats_dir = cfg.TEST.CLASSIFIED_STATS_DIR
    groundtruth_stats_dir = cfg.TEST.GROUNDTRUTH_STATS_DIR
    classified_image_dir = cfg.TEST.CLASSIFIED_IMAGE_DIR

    if os.path.exists(classified_stats_dir): shutil.rmtree(classified_stats_dir)
    if os.path.exists(groundtruth_stats_dir): shutil.rmtree(groundtruth_stats_dir)
    if os.path.exists(classified_image_dir): shutil.rmtree(classified_image_dir )

    os.makedirs(classified_stats_dir)
    os.makedirs(groundtruth_stats_dir)
    os.makedirs(classified_image_dir)

    # Build Model
    input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
    feature_maps = YOLOv3(input_layer, NUM_CLASS)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i, NUM_CLASS)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.load_weights(cfg.TRAIN.WEIGHTS_DIR+'Y3')

    with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # gives blue filter
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])
            test_filename = line.split(" ")[0]
            test_bboxes_gt = line.split(" ")[1:]


            # delete all entrys of current file from traing_results
            if not df_map is None:
                df_map.drop(df_map[df_map.IMAGE == test_filename].index,inplace=True)


            if len(bbox_data_gt) == 0:
                bboxes_gt=[]
                classes_gt=[]
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(groundtruth_stats_dir, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([image_path, class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())


            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(classified_stats_dir, image_name[:-4] + '.txt')
            # Predict Process
            image_size = image.shape[:2]
            image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            pred_bbox = model.predict(image_data)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
            bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')

            if cfg.TEST.CLASSIFIED_IMAGE_DIR is not None:
                image = utils.draw_bbox(image, bboxes)
                cv2.imwrite(classified_image_dir+'/'+image_name, image)

            with open(predict_result_path, 'w') as f:
                for bbox in bboxes:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([image_path, class_name, xmin, ymin, xmax, ymax, score]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
                     

                    # save metrics
                    test_bbox_gt_class  = [ np.array(x.split(',')[:4], dtype=np.int) \
                            for x in test_bboxes_gt if int(x.split(',')[4]) == bbox[5] ]
                    test_bbox_gt_class_len = len(test_bbox_gt_class)
                    test_bbox_class  = [ x for x in bboxes if x[5] == bbox[5] ]
                    # bboxes per image from annotations
                    test_bbox_class_len = len(test_bbox_class )
                    # compute max iou for current clas
                    max_iou_class = max( [ utils.bboxes_iou(bb,bbox[:4]) for bb in test_bbox_gt_class ] or [0]     )


                    df_map = df_map.append({
                        'IMAGE':test_filename, # IMAGE
                        'CLASS':class_name , # CÖASS
                        'SCORE':bbox[4], # SCORE
                        'IOU':max_iou_class, # IOU
                        'BBOXPR':test_bbox_class_len, #BBOXPR
                        'BBOXGT':test_bbox_gt_class_len }, ignore_index=True )    # BBOX

    if df_map.size > 0:

        p = os.path.join(cfg.TRAIN.METRICS_DIR,prefix_filename)
        if not os.path.exists(p): os.makedirs(p)

        df_map = utils.preprocess_map(df_map, tp_th=0.5)

        utils.plot_map(df_map, 
            p+'/epoch'+str(epoch+1)+'_test_mAP.png',
            10,
            NUM_CLASS 
            )
        df_map.to_csv(p+'/epoch'+str(epoch+1)+'_test_mAP.csv')


    # save model with trained weights after each epoch
    if args.export:
        #tf.keras.models.save_model(
        tf.saved_model.save(
            model,
            os.path.join(cfg.TRAIN.EXPORTS_DIR,prefix_filename,'1')
            #overwrite=True
            #include_optimizer=True,
            #save_format=None,
            #signatures=None,
            #options=None
           )
