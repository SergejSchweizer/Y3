# YOLOv3 Tensorflow2 (Keras)
---
### Training and evaluation of 600 Classes from Open Images Dataset V5 with transferlearning, freeze layers, gpu
##### Author: Sergej Schweizer (SSC)
##### The [original yolo3 code](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3) was rewritten and testet with several thousands of classes.
##### Finally this work ended in: https://github.com/SergejSchweizer/Y3

#### 0. Install necessary packages
---

Current dependencies for tensorflow version: 
* cuda           = 10.0
* cudnjn         = 7.6.4
* tensorflow-gpu = 2.2

```python
!pip install pandarallel tensorflow-gpu==2.2
```

#### 1. Import libarays
---


```python
import os
import pandas as pd
import numpy as np
import scipy
import random
import easydict
import tensorflow
import PIL
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandarallel
import warnings
warnings.filterwarnings("ignore")
```

#### 2. Y3 working path.
---

Withhin this path all the data will be stored in subdirectorys.
E.g: 
* annotations - subdir:conf
* classified images - subdir:images
* model exports - subdir:exports
* computed weights - subidr:weights
* computerd mAP - subdir:metrics

We need to create at least conf and images subdirectorys, other subdirectorys will be created depnding on the trainig options of the YOLO3 model


```python
train_dir = '/exports/DATA/Y3_openimageV5/'
code_dir = '/home/sergej/'
#train_dir = 'Y3_openimageV5/'
#code_dir = 'Y3'
for p in ['conf','images']:
    if not os.path.exists(train_dir+'/'+p):
        os.makedirs(train_dir+'/'+p, exist_ok=True)
```


```python
%cd $train_dir'conf'
```

#### 3. Download annotatoins, labels and images (this can take a while)
---


```python
!wget -q https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv
```


```python
!wget -q https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
```


```python
!wget -q https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/test.zip
```

#### 4. Unzip images
---


```python
!unzip -qq test.zip
```

#### 5. Create annototations and classes dataframes
---
##### 5.1 Read csv to dataframe


```python
raw_annots = pd.read_csv("test-annotations-bbox.csv")
raw_classes = pd.read_csv("class-descriptions-boxable.csv", names=["name","label"])
```

##### 5.2 Create classes series


```python
classes = raw_classes.label
```

#### 6. Add new columns for better preprocessing
---
Functions for transfering annot data


```python
def get_index_of_class(x):
    return  raw_classes[ raw_classes.name == str(x) ].index[0]
```


```python
def get_label_of_class(x):
    return  raw_classes[ raw_classes.name == str(x) ].label.values[0]
```


```python
imgs = {}
def get_width_of_img(img):
    if not img in imgs:
        imgs[img]=PIL.Image.open(img).size
    return imgs[img][0]
  
def get_height_of_img(img):
    if not img in imgs:
        imgs[img]=PIL.Image.open(img).size
    return imgs[img][1]
```


```python
annots = raw_annots[['ImageID', 'XMin', 'YMin', 'XMax', 'YMax','LabelName']]#[:100]   # slice the to lower size for testing
```


```python
annots.loc[:,'LabelIndex'] = annots.loc[:,'LabelName'].apply(get_index_of_class)
annots.loc[:,'Label'] = annots.loc[:,'LabelName'].apply(get_label_of_class)
annots.loc[:,'ImagePath'] = train_dir+'test/'+annots.loc[:,'ImageID']+'.jpg'
annots.loc[:,'width'] = annots.loc[:,'ImagePath'].apply(get_width_of_img)
annots.loc[:,'height'] = annots.loc[:,'ImagePath'].apply(get_height_of_img)
```


```python
annots.loc[:,'XMin'] = annots.loc[:,'XMin'].multiply(annots['width']).astype(int)#.astype('str')+','
annots.loc[:,'XMax'] = annots.loc[:,'XMax'].multiply(annots['width']).astype(int)#.astype('str')+','
annots.loc[:,'YMin'] = annots.loc[:,'YMin'].multiply(annots['height']).astype(int)#.astype('str')+','
annots.loc[:,'YMax'] = annots.loc[:,'YMax'].multiply(annots['height']).astype(int)#.astype('str')+','
annots['polygon'] = annots.loc[:,['XMin','YMin','XMax','YMax']].values.tolist()
```

#### 7. Check if classes are imbalanced
---


```python
annots.Label.value_counts().plot.bar(figsize=(35,20))
```


```python
annots.Label.value_counts()
```

#### 8. Create Annotations
---


```python
def create_annots(annots, classes):
    
    _library_prefix='/exports/LIBRARY/Library'
    annotations = []


    for i in annots.ImageID.unique():
     
        full_image_path = annots[annots['ImageID'] == i].iloc[0]['ImagePath']
            
        # if filename does not exitst, continue
        if not os.path.exists(full_image_path):
            print('ERROR: Image does not exist: {}'.format(full_image_path))
            continue
    
        labels = []
    
        # for same image
        for n in range(len(annots[annots['ImageID'] == i ])):
        
            x_min,y_min,x_max,y_max = [ i for i in annots[annots['ImageID'] == i].iloc[n]['polygon'] ]
            width = annots[annots['ImageID'] == i].iloc[n]['width'].astype(int)
            height = annots[annots['ImageID'] == i].iloc[n]['height'].astype(int)
                  
            # check if second x is higher than width or first y is higher than height
            if (( x_max > width) or (y_max > height )):
                print('ERROR: x:{} is higher than widht:{} or y:{} is higher than height:{}'.format(x_max, width, y_max, height))
                continue
            
            labels.append('{},{},{},{},{}'.format(x_min, y_min, x_max, y_max, annots[annots['ImageID'] == i].iloc[n]['LabelIndex']))
            #print(_library_prefix+srclocation+'/'+srcname)
           
        # ommit if no labes for current image
        if len(labels) == 0:
            continue
        # append image with labes to annots    
        annotations.append(full_image_path+' '+' '.join(labels))   

    return annotations
```


```python
annotations = create_annots(annots, raw_classes)
```

#### 9. Shuffle and split annotations into train and test parts
---
All files will be saved to the Y3 working directory (in the conf subdirectory)
Annotations will be splitted in train.txt (95%) and test.txt(5%) files for training and testing.


```python
def save_list_to_file(file_,list_):
    annot_file = open(file_, "w")

    # list in list
    if isinstance(list_[0],list):
        for path,polygon in list_:
            annot_file.write(str(path) + ' ' + str(polygon) + '\n')
            
    # list
    if isinstance(list_[0],str):
        for full_string in list_:
            annot_file.write(str(full_string) + '\n')
          
    annot_file.close()
```


```python
def shuffle_and_split(annots,test=5):
    test = 5  # % for testing
    random.shuffle(annots)
    l = len(annots)
    #print(l)
    
    training_list = annots[ : int(l*((100-test)/100)) ]
    test_list = annots[ int(l*((100-test)/100)) : -1 ]


    save_list_to_file('train.txt',training_list)
    save_list_to_file('test.txt',test_list)

    !sed s/"jpg,"/"jpg "/g -i *.txt
```


```python
shuffle_and_split(annotations,5)
```

#### 10. (optional) Draw labels to images, to check if our labels are sized properly
---
It make sense run this process once at the beginning to check whether the labels are set correct


```python
def draw_rectangels_to_labeld_images(annot_file,save_path, relative_path=False):

    FIG_SIZE=(30,15)
    
    annot_file = open(annot_file, "r")
    annotations = annot_file.read().splitlines()

    # every annotation is leafleat
    for l in annotations:
       
        filename = l.split()[0].split('/')[-1][:-3]+'png'
        pathfilename = l.split()[0]
        
        if relative_path:
            save_path = save_path+os.path.dirname(pathfilename)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        else:
            save_filepath = save_path
            
        polygons =  [ int(y) for x in str(l.split('jpg ')[1]).split(',') for y in x.split(' ') ]
                
        im = PIL.Image.open(pathfilename)
        
        # Display the image
        plt.figure(figsize = FIG_SIZE)
        plt.imshow(im, interpolation='nearest')

        # Get the current reference
        ax = plt.gca()

        #for polygon in [ polygons[i:i+8] for i in range(0,len(polygons),8) ]:
        for pol in [ polygons[i:i+5] for i in range(0,len(polygons),5) ]:
            #pol = [ int(p) for p in pol]
            #print(pol)
            rect = Rectangle( (pol[0], pol[1]), int(pol[2]-pol[0]), int(pol[3]-pol[1]),  linewidth=3,edgecolor='r',facecolor='none')
            ax.text(pol[0], pol[1], classes[pol[4]], color='red')
            ax.add_patch(rect)
    
        plt.savefig(save_path+'/'+filename, bbox_inches='tight')
        plt.close('all')
        im.close()
```


```python
train_dir = '/exports/DATA/Y3_openimageV5/'
draw_rectangels_to_labeld_images(
            train_dir+'/conf/test.txt'
            ,train_dir+'/images/labeled'
            ,relative_path=False)
```

#### 11. Prepare YOLO3 Framework
---
##### 11.1 Checkout YOLO3 Version


```python
%cd $code_dir
!git clone https://github.com/SergejSchweizer/Y3.git
```

##### 11.2 Install dependencys
You need root permissions to install packages


```python
!pip install -r Y3/requirements.txt
```

##### 11.3 Download YOLO3 weights from Joseph Redmons Page (May the power....)


```python
!wget -q https://pjreddie.com/media/files/yolov3.weights
```

##### 11.3 Y3 Command line options
* --warmupepochs:  the number of initial epochs during which the sizes of the 5 boxes in each cell is forced to match the sizes of the 5 anchors, this trick seems to improve precision emperically
* --epochs: number of epochs
* --map: create mean overage precision csv and png files
* --batchsize: how many images should be computed at once
* --gpu:  use gpu, you can define amount of MB which should be used at the gpu device
* --trainpath: path to Y3 directory, which should consist at least of conf (including train.txt,test.txt and classes.names) and images subdirectory
* --freezebody: 1 - no, 2 = only darknet, 3 = all except 3 last layers
* --exports: export weights for using in rest-api
* --transferlearing: define weights file, which will be loaded at the beginning


```python
!python Y3/Y3.py --warmupepochs 2 --epochs 5 --map --batchsize 5  --trainpath $train_dir --transferlearning yolov3.weights --freezebody 1 --export
```

#### 12. Mean average precision
---
The performance of object localization and classification is measured through the Mean Average Precison. This current mAP chart (7 Epochs, with transferlearning) consists of 10 best and worst classes according to their AUC (area under curve) value. The red line shows the average performance.

Following statements can be made based on the mAP:
* the more annotatins of certain class - the better performance
* also the label quality plays a role (e.g. Vehicle has 1080 annots, but mAP is only 0.029)
* use only classes with at least 300 annots.
* check whether labels are set correct
* improve data-augmentation
* check failure summs for every image (remove outliers)
![Mean Average Precision](https://github.com/SergejSchweizer/Y3/blob/master/mAP/OpenImageV5/epoch7_test_mAP.png?raw=true)

```python

```
