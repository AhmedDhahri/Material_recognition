# Material_recognition

**All scripts must be executed from outside the main directory**

#### 1- Download datasets:
Using shell script get_datasets.sh allows to download datasets (IRH, MINC, SUN, EPFL):

````
sh Material_recognition/get_datasets.sh
````
#### 2- Training
Every training phase is implemented on a separate script.

###### 2.1- Train on MINC large-scale dataset
````
python3 Material_recognition/train/train_minc.py swinv2b False
````


###### 2.2- Train on SUN-EPFL Unlabeled datasets
Contrastive Visual Representation Learning (CVRL)
````
python3 Material_recognition/train/train_minc.py swinv2b False
````


###### 2.3- Train on IRH multi-modal dataset

````
python3 Material_recognition/train/train_minc.py swinv2b False
````

#### 3- Tests and inference

