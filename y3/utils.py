#! /usr/bin/env python
# coding=utf-8

import cv2
import random
import colorsys
import numpy as np
from y3.config import cfg
from pandarallel import pandarallel
import matplotlib.pyplot as plt

pandarallel.initialize()

YOLO_V3_LAYERS = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
    ]

def preprocess_map(df_map, tp_th=0.5):

    if not df_map.size == 0:
        def gtt(x):
            return int(x > tp_th) 
        def ltt(x):
            return int(x < tp_th) 
                                                    
        df_map = df_map.sort_values('SCORE', ascending=False)
                                                                
        # TP,FP
        df_map['TP']  = df_map['IOU'].parallel_map(gtt)
        df_map['FP']  = df_map['IOU'].parallel_map(ltt)
                                                                                            
        # CUM TP
        df_map['CUM_TP'] = df_map['TP'].cumsum()
        df_map['CUM_FP'] = df_map['FP'].cumsum()
                                                                                                              # bboxes are per image and class from annotations files
        df_map['CUM_GT']    = df_map.drop_duplicates(subset = ["IMAGE","CLASS"])['BBOXGT'].sum()

        df_map['P_ALL']  = df_map['CUM_TP'] / ( df_map['CUM_TP'] + df_map['CUM_FP'] )
        df_map['R_ALL']  = df_map['CUM_TP'] / df_map['CUM_GT'] 
        
        df_map['AUC_ALL'] = np.trapz(df_map['P_ALL'],  df_map['R_ALL'])

        df_map['CUM_TP_CLASS' ] = df_map.groupby(['CLASS'])['TP'].cumsum()
        df_map['CUM_FP_CLASS' ] = df_map.groupby(['CLASS'])['FP'].cumsum()

        for class_ in df_map.CLASS.unique():
            # CUM_GT_CLASS (gt)
            df_map.loc[ (df_map.CLASS == class_, 'CUM_GT_CLASS')] = \
            df_map.loc[ (df_map.CLASS == class_)].drop_duplicates(subset = ["IMAGE","CLASS"])['BBOXGT'].sum()

            # P_CLASS
            df_map.loc[ (df_map.CLASS == class_, 'P_CLASS')] = \
            df_map.loc[ (df_map.CLASS == class_, 'CUM_TP_CLASS') ] / \
            ( df_map.loc[ (df_map.CLASS == class_, 'CUM_TP_CLASS') ] + \
            df_map.loc[ (df_map.CLASS == class_, 'CUM_FP_CLASS') ] )

            # R_CLASS
            df_map.loc[ (df_map.CLASS == class_, 'R_CLASS')] = \
            df_map.loc[ (df_map.CLASS == class_, 'CUM_TP_CLASS') ] / \
            df_map.loc[ (df_map.CLASS == class_, 'CUM_GT_CLASS')]

            df_map.loc[ (df_map.CLASS == class_, 'AUC_CLASS')] = \
            np.trapz( df_map.loc[ (df_map.CLASS == class_, 'P_CLASS')],\
            df_map.loc[ (df_map.CLASS == class_, 'R_CLASS')] )

        df_map = df_map.sort_values('AUC_CLASS', ascending=False)
        return df_map

def plot_map(df_map, path, num_classes, uniq_classes ):

    if df_map is None:
        print("df_map is None")
        return   
 
    
    if ( 'R_ALL' in df_map.columns\
        and 'P_ALL' in df_map.columns \
        and len(df_map.index.values) != 0 ):

    
        df_map = df_map.sort_values('AUC_CLASS', ascending=False)
	    
        legends_list = []
        classes_list = []
        classes_list.extend(list(df_map[df_map['AUC_CLASS'] > 0].CLASS.unique()[:num_classes]))
        classes_list.extend(list(df_map[df_map['AUC_CLASS'] > 0].CLASS.unique()[-num_classes:]))
        classes_list = sorted(set(classes_list), key=lambda x: classes_list.index(x))
	    
        df_map = df_map.sort_values('SCORE', ascending=False)
	    
        plt.figure(figsize=(25,15))
        plt.plot('R_ALL'
        ,'P_ALL'
        ,'r-o'
        ,data=df_map
        ,linewidth=0.1
        ,markersize=2
        )
	    
        legends_list.extend(["class: %s, AUC: %.3f P: %.3f, R: %.3f, C: %s" % ('all'
            ,df_map['AUC_ALL'].iloc[-1] 
            ,df_map['P_ALL'].iloc[-1]
            ,df_map['R_ALL'].iloc[-1]
            ,df_map.size

                          )]
             )
	      
        for class_ in classes_list:
            if df_map.loc[ (df_map.CLASS == class_, 'AUC_CLASS')].sum() > 0:
                plt.plot(
                df_map[ df_map['CLASS'] == class_]['R_CLASS'].values
                ,df_map[ df_map['CLASS'] == class_]['P_CLASS'].values
                ,'-o'
                ,label = class_
                ,linewidth=0.1
                ,markersize=2
                )

                legends_list.extend(["class: %s, AUC: %.3f P: %.3f, R: %.3f, C: %s" % ( class_
                    ,df_map[ df_map['CLASS'] == class_]['AUC_CLASS'].iloc[-1] 
                    ,df_map[ df_map['CLASS'] == class_]['P_CLASS'].iloc[-1]
                    ,df_map[ df_map['CLASS'] == class_]['R_CLASS'].iloc[-1]
                    ,df_map[ df_map['CLASS'] == class_].size 
                    )]
                )
		    
        plt.title('mean Average Precision. ({} best and worst classes of {} based on AUC)'.format(len(classes_list),uniq_classes))        
        plt.legend(legends_list)        
        plt.grid(True)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        #plt.show()
        plt.savefig(path)
        plt.close()



def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    layers = YOLO_V3_LAYERS

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
              sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)

                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))

            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read weigts'
    wf.close()


def load_weights(model, weights_file):
    """
    I agree that this code is very ugly, but I don’t know any better way of doing it.
    """
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    #first_conv2d_layer_number=model.layers[1].name
    
    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        #print(model.layers[1].name)
        #print(conv_layer_name)
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [58, 66, 74]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
        #print(conv_weights)
        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preporcess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    #colors = list(map(lambda x: (int(x[0] * 55), int(x[1] * 55), int(x[2] * 55)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.7
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)

            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image



def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area + 1e-10
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []
    ious = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            # get highest confidense level
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            
            # (xmin, ymin, xmax, ymax, score, class)
            #[3.97346771e+02 1.42703333e+03 5.90519836e+02 1.56839392e+03 9.74209785e-01 0.00000000e+00]
            # best_bbox

            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])

            #print('CLASS: {}, IOU: {} '.format(cls, iou))

            #best_iou  = bboxes_iou(best_bbox[np.newaxis, :4], labels_bboxes[:, :4])
            #best_iou = best_iou[np.argmax(best_iou)]
            #ious.append(best_iou)

            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0


            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

