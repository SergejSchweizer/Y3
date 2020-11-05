# YOLOv3 Tensorflow2-gpu training and evaluation on 600 Classes from Open Images Dataset V5
### with transferlearning, freeze layers, gpu
##### Author: Sergej Schweizer (SSC)
##### The [original yolo3 code](https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3) was rewritten and testet with several thousands of classes.
##### Finally this work ended in: https://github.com/SergejSchweizer/Y3


```python
# The current tensorflow version depends on:
#cuda           = 10.0
#cudnjn         = 7.6.4
#tensorflow-gpu = 2.0
```


```python
#!pip install pandarallel tensorflow-gpu==2.0
```

### Import libraries



```python
import os
import pandas as pd
import numpy as np
import scipy
import easydict
import tensorflow
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandarallel
```

# Prepare Open Images Dataset V5 (Google)

### Download Images, Bonding Boxes and Classe Names (take few minutes)


```python
!wget -q https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/test.zip
!wget -q https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv
!wget -q https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
```

### Unzip image archive


```python
!unzip -qq test.zip
```

### Create YOLO3 directory structure


```python
train_dir = 'YOLO3_SSC'
for p in ['conf','images']:
    if not os.path.exists(train_dir+'/'+p):
        os.makedirs(train_dir+'/'+p, exist_ok=True)
```

### Read csv to dataframe


```python
raw_annots = pd.read_csv("test-annotations-bbox.csv")
raw_classes = pd.read_csv("class-descriptions-boxable.csv", names=["name","label"])
```

### Get data from raw_annotations (you can limit it by the slicing at the and)


```python
annots = raw_annots[['ImageID', 'XMin', 'YMin', 'XMax', 'YMax','LabelName']]#[:100]
```

### Function for transfering annot data


```python
def get_index_of_class(x):
    return  raw_classes[ raw_classes.name == str(x) ].index[0]
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

### Add addtional collumn with label index (take few minutes)


```python
annots.loc[:,'LabelIndex'] = annots.loc[:,'LabelName'].apply(get_index_of_class)
```


```python
annots.loc[:,'ImagePath'] = 'test/'+annots.loc[:,'ImageID']+'.jpg'
```


```python
annots.loc[:,'width'] = annots.loc[:,'ImagePath'].apply(get_width_of_img)
annots.loc[:,'height'] = annots.loc[:,'ImagePath'].apply(get_height_of_img)
```


```python
annots.loc[:,'XMin'] = annots['XMin'].multiply(annots['width']).astype(int)#.astype('str')+','
annots.loc[:,'XMax'] = annots['XMax'].multiply(annots['width']).astype(int)#.astype('str')+','
annots.loc[:,'YMin'] = annots['YMin'].multiply(annots['height']).astype(int)#.astype('str')+','
annots.loc[:,'YMax'] = annots['YMax'].multiply(annots['height']).astype(int)#.astype('str')+','
```

#### Save all annots


```python
annots = annots.loc[:,['ImagePath', 'XMin', 'YMin', 'XMax', 'YMax','LabelIndex']]
annots.to_csv('annots.csv',index=False, header=False, sep=',')
```

### Save clases.csv


```python
raw_classes.label.to_csv(train_dir+'/conf/classes.names',index=False, header=False)
classes = raw_classes.label
```


```python
test = 5  # % for testing
annots = annots.sample(frac=1).reset_index(drop=True)
l = len(annots)
       
df_training = annots[ : int(l*((100-test)/100)) ]
df_test = annots[ int(l*((100-test)/100)) : -1 ]

df_training.to_csv(train_dir+'/conf/train.txt',index=False, header=False,sep=',')
df_test.to_csv(train_dir+'/conf/test.txt',index=False, header=False,sep=',')

!sed s/"jpg,"/"jpg "/g -i YOLO3_SSC/conf/*.txt

```

### Draw labels to images, to check if our labels are sized properly


```python
def draw_rectangels_to_labeld_images(annot_file,save_path, relative_path=False):

    FIG_SIZE=(30,15)
    
    annot_file = open(annot_file, "r")
    annotations = annot_file.read().splitlines()

    # every annotation is leafleat
    for l in annotations:
        #print(l)
        filename = l.split()[0].split('/')[-1][:-3]+'png'
        pathfilename = l.split()[0]
        
        #if relative_path:
        if relative_path:
            save_filepath = save_path+os.path.dirname(pathfilename)
        else:
            save_filepath = save_path

        if not os.path.exists(save_filepath):
            os.makedirs(save_filepath)
           
      
        polygons =  [ int(y) for x in str(l.split('jpg ')[1]).split(',') for y in x.split(' ') ] 
       
        im = Image.open(pathfilename)

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
    
        plt.savefig(save_filepath+'/'+filename, bbox_inches='tight')
        plt.close('all')
        im.close()
```


```python
draw_rectangels_to_labeld_images(
            train_dir+'/conf/test.txt'
            ,train_dir+'/images/labeled'
            ,relative_path=False)
```

# Prepare YOLO3 Framework

### Checkout YOLO3 Version


```python
! git clone https://github.com/SergejSchweizer/Y3.git
```

### Download YOLO3 weights from Joseph Redmons Page (May the power....)


```python
!wget https://pjreddie.com/media/files/yolov3.weights
```

#### Y3 Command line options
###### --warmupepochs:  the number of initial epochs during which the sizes of the 5 boxes in each cell is forced to match the sizes of the 5 anchors, this trick seems to improve precision emperically
###### --epochs: number of epochs
###### --map: create mean overage precision csv and png files
###### --batchsize: how many images should be computed at once
###### --gpu:  use gpu, you can define amount of MB which should be used at the gpu device
###### --trainpath: path to Y3 directory, which should consist at least of conf (including their content) and images subdirectory
###### --freezebody: 1 - no, 2 = only darknet, 3 = all except 3 last layers
###### --exports: export weights for using in rest-api
###### --transferlearing: define weights file, which will be loaded at the beginning


```python
 !python Y3/Y3.py --warmupepochs 2 --epochs 5 --gpu 10000 --map --batchsize 5  --trainpath YOLO3_SSC --transferlearning yolov3.weights --freezebody 1 --export
```

    => EPOCH:1 STEP 3453   lr: 0.000001   giou_loss: 2.1011   conf_loss: 656.1710   prob_loss: 483.7092   total_loss: 1141.9812 
    => EPOCH:1 STEP 3454   lr: 0.000001   giou_loss: 2.8979   conf_loss: 664.5010   prob_loss: 613.9684   total_loss: 1281.3674 
    => EPOCH:1 STEP 3455   lr: 0.000001   giou_loss: 2.0210   conf_loss: 655.8571   prob_loss: 379.4922   total_loss: 1037.3704 
    => EPOCH:1 STEP 3456   lr: 0.000001   giou_loss: 3.9775   conf_loss: 660.3450   prob_loss: 1137.1315   total_loss: 1801.4539 
    => EPOCH:1 STEP 3457   lr: 0.000001   giou_loss: 3.2536   conf_loss: 660.4791   prob_loss: 806.0881   total_loss: 1469.8208 
    => EPOCH:1 STEP 3458   lr: 0.000001   giou_loss: 2.0572   conf_loss: 659.2932   prob_loss: 405.3122   total_loss: 1066.6626 
    => EPOCH:1 STEP 3459   lr: 0.000001   giou_loss: 2.8047   conf_loss: 659.5377   prob_loss: 566.0484   total_loss: 1228.3909 


### Mean overage Precision


```python

```
