# Material_recognition

**All scripts must be executed from outside the main directory**
**Possible MODEL_NAME are: swinv2b, vith14, eva02l14, maxvitxl, coatnet2. moat4**
**Possible EXPERIMENT values are 0 for RGB training, 1 for RGB-NIR training and 2 for DEPTH training.**

#### 1- Download datasets:
Using a shell script *get_datasets.sh* allows to download datasets (IRH, MINC, SUN, EPFL), automatically, and place each one in its adequate directory. Use this commmand line to run the script.

````
sh Material_recognition/get_datasets.sh
````
#### 2- Training
Every training phase is implemented on a separate script.

###### 2.1- Train on MINC large-scale dataset
Run training for 5 epochs on MINC large-scale dataset.  
````
python3 Material_recognition/train/train_minc.py MODEL_NAME LOAD
````

- Replace```MODEL_NAME``` with the desired model from the list above.

- Replace```LOAD``` with {True, False}. True means the model would load the saved checkpoint, while false means the model would load default backbone pre-trained weights. 



###### 2.2- Train on SUN-EPFL Unlabeled datasets
Run Contrastive Visual Representation Learning (CVRL) to learn intra-modal and inter-modal visual representations between RGB and DEPTH as well as RGB and NIR.

````
python3 Material_recognition/train/train_irh.py MODEL_NAME EXPERIMENT
````
- Replace ```MODEL_NAME``` with the desired model from the list above.
- Replace ```EXPERIMENT``` with the number of fused backbones.


###### 2.3- Train on IRH multi-modal dataset

````
python3 Material_recognition/train/train_irh.py MODEL_NAME BATCH_SIZE EPOCHS EXPERIMENT
````
- Replace ```MODEL_NAME```  with the desired model from the list above.
- Replace ```BATCH_SIZE``` with the adequate batch size.
- Replace ```EPOCHS```  with the desired epochs.
- Replace ```EXPERIMENT```  with the number of fused backbones.

#### 3- Tests and inference
Testing models and computing metrics.

###### 3.1- Inference on an image
Run inference on an input image file(s) and get the predicted class name.

````
python3 Material_recognition/test/inference.py MODEL_NAME EXPERIMEMT FILE1 FILE2 FILE3
````
- Replace ```MODEL_NAME```  with the desired model from the list above.
- Replace ```EXPERIMENT```  with the number of fused backbones.
- Replace ```FILE<>```  with the list of paths of input images. ```FILE1```, ```FILE2```, ```FILE3``` are respectively for RGB, NIR and DEPTH input images. ```FILE2``` and ```FILE3``` are optional. 

###### 3.2- Test TOP-1 accuracy and TOP-5 accuracy on MINC dataset
Compute the TOP-1 accuracy for a desired model.

````
python3 Material_recognition/test/accuracy_test_minc.py MODEL_NAME NUM_WORKERS
````
- Replace ```MODEL_NAME```  with the desired model from the list above.
- Replace ```NUM_WORKERS```  with the desired number of loading processes.

###### 3.3- Test TOP-1 accuracy on IRH and MINC dataset
Compute the TOP-1 accuracy of a desired model on IRH dataset.

````
python3 Material_recognition/test/accuracy_test_irh.py MODEL_NAME EXPERIMENT NUM_WORKERS BATCH_SIZE
````
- Replace ```MODEL_NAME```  with the desired model from the list above.
- Replace ```EXPERIMENT```  with the number of fused backbones.
- Replace ```NUM_WORKERS```  with the desired number of loading processes.
- Replace ```BATCH_SIZE``` with the adequate batch size.


Compute the TOP-1 accuracy of a desired model and a desired dataset. With the possibility to select a specific classe.
````
python3 Material_recognition/test/test.py MODEL_NAME CLS EXPERIEMT
````
- Replace ```MODEL_NAME```  with the desired model from the list above.
- Replace ```CLS```  with the desired data class.
- Replace ```EXPERIMENT```  with the number of fused backbones. **Here there is an exception. Experiment could be 3. By selecting 3 MINC dataset would be loaded.**

###### 3.4- Run zero-shot test on IRH dataset
Compute the zero-shot accuracy on IRH dataset.
````
python3 Material_recognition/test/test_zero_shot.py MODEL_NAME
````
- Replace ```MODEL_NAME```  with the desired model from the list above.

###### 3.5- Run CRF-based segmentation on a scene photo from IRH dataset
Visualize segmentation capabilities of the models. This method uses the average of multi-scale prediction followed by a CRF layer.

````
python3 Material_recognition/test/crf/vis_crf.py MODEL_NAME IMG_ID EXPERIEMT
````

- Replace ```MODEL_NAME```  with the desired model from the list above.
- Replace ```IMG_ID```  with the image id (6 digit number). Using the ID of the adequate RGB, NIR and DEPTH images would be loaded.
- Replace ```EXPERIMENT```  with the number of fused backbones.