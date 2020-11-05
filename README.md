# YOLO3 training on 600 Classes from Open Images Dataset V6
##### SSC: Sergej Schweizer 04.11.2020
##### The original yolo3 code was rewritten and testet with several thousand of classes.
##### Finally this work ended in: https://github.com/SergejSchweizer/Y3


```python
!pip install pandarallel
```

### Import librarys



```python
import os
import pandas as pd
import numpy as np
#import pillow
import scipy
#import wget
import seaborn
import easydict
#import grpcio
import tensorflow
import PIL
import pandarallel

```

# Prepare Open Images Dataset V6 (Google)

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
annots['LabelIndex'] = annots['LabelName'].apply(get_index_of_class)
```


```python
annots['ImagePath'] = 'validation/'+annots['ImageID']+'.jpg'
```


```python
annots['width'] = annots['ImagePath'].apply(get_width_of_img)
annots['height'] = annots['ImagePath'].apply(get_height_of_img)
```


```python
annots['XMin'] = annots['XMin'].multiply(annots['width']).astype(int)#.astype('str')+','
annots['XMax'] = annots['XMax'].multiply(annots['width']).astype(int)#.astype('str')+','
annots['YMin'] = annots['YMin'].multiply(annots['height']).astype(int)#.astype('str')+','
annots['YMax'] = annots['YMax'].multiply(annots['height']).astype(int)#.astype('str')+','
```

#### Save all annots


```python
annots = annots.loc[:,['ImagePath', 'XMin', 'YMin', 'XMax', 'YMax','LabelIndex']]
annots.to_csv('annots.csv',index=False, header=False, sep=',')
```


```python
annots
```

### Save clases.csv


```python
 raw_classes.label.to_csv(train_dir+'/conf/classes.names',index=False, header=False)
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


```python

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

### Mean overage Precision


```python

```

