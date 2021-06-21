# Improved detection of small objects in road sequences using convolutional neural networks and super-resolution

In this repository, a new proposal has been developed. It is focused on using a fine adjustment of the network (it does not increase the execution time) that allows it to be automatically adapted to the traffic scene without human intervention. Firstly, we propose to apply a super resolution algorithm to detect objects in the scene that would otherwise go unnoticed by the DCNN object detection method. These detected objects or labelled data are used to generate a training dataset that is used to tune the network. All this process is done offline and only once per scene. Once the training is finished, the network has been adapted to better detect objects with the given camera distance and perspective.


To implement this new functionality, denoising and super-resolution processes have been applied as a pre-processing step. The implementation of this proposal is in the following directory:

* https://github.com/IvanGarcia7/SR-FT_ENHANCEMENT/blob/main/TFM_codigo.ipynb

Within the jupyter notebook, all the necessary steps are established to initialize the work environment, download the pre-trained super-resolution and object detection models and carry out the proposal. For this purpose, a function has been developed that automatically performs the pre-processing phase and subsequent element detection, denoted as make_inference_SR. This function takes as input the image on which you want to detect the elements, the name under which the image with the generated detections will be stored and finally the directory in which you want to store it.


# Workflow of the proposed technique:

![WORKFLOW](https://github.com/IvanGarcia7/SR-FT_ENHANCEMENT/blob/main/IMAGES/Proposal.png?raw=true)


The Offline part is composed of the SR application, in addition to the generation of the dataset for Fine-Tuning. The Online part is composed of the detection performed by the retrained model.


# Execution Test:

## Input Image:

After downloading and loading the necessary models, we want to improve the detections on a given image as input, e.g. the Figure below:

![INPUT IMAGE](https://github.com/IvanGarcia7/SR-FT_ENHANCEMENT/blob/main/IMAGES/Input.jpg?raw=true “INPUT IMAGE”)

Subsequently, it is necessary to execute the following function:

``` 
make_inference_SR(‘/usr/share/Data2/small_objects/Evaluation/1.jpg’, ‘1_output.jpg’, ‘/usr/share/Data2/small_objects/SR_OUTPUT/‘, 0)
```

After performing the successive detections, an output image will be generated. Using the values determined by the function, an XML file is subsequently generated, which is necessary to generate the Tensorflow 2 training records.

## Output:

![OUTPUT IMAGE](https://github.com/IvanGarcia7/SR-FT_ENHANCEMENT/blob/main/IMAGES/Output.jpg?raw=true “OUTPUT IMAGE”)


# Fine-Tuning:

To carry out the generation of the records, it will be necessary to execute the following command after separating the images intended for testing and training.

``` 
!python '/usr/share/Data2/small_objects/Retrain/Scripts/generate_tfrecord.py' -x /usr/share/Data2/small_object/Retrain/TRAIN/train -l /usr/share/Data2/small_object/Retrain/LABELMAP/label_map.pbtxt -o /usr/share/Data2/small_object/Retrain/TRAIN/RECORD-TRAIN/train.record

!python '/usr/share/Data2/small_objects/Retrain/Scripts/generate_tfrecord.py' -x /usr/share/Data2/small_object/Retrain/VALIDATION/train -l /usr/share/Data2/small_object/VALIDATION/LABELMAP/label_map.pbtxt -o /usr/share/Data2/small_object/Retrain/VALIDATION/RECORD-TRAIN/validation.record
```

Finally, it only remains to run the script to perform the Fine-Tuning after setting the hyper-parameters in the configuration file.

``` 
!python /usr/share/Data2/small_objects/Retrain/SCRIPTS/model_main_tf2.py --model_dir=/usr/share/Data2/small_objects/Retrain/myEfficient --pipeline_config_path=/usr/share/Data2/small_objects/Retrain/myEfficient/ssd_efficientdet_d4_1024x1024_coco17_tpu-32.config

```

The scripts mentioned in the Fine-Tuning phase can be found in the following path:

* https://github.com/IvanGarcia7/SR-FT_ENHANCEMENT/tree/main/SCRIPTS


# Evaluation:

To evaluate the mAP(Mean Average Precision) of the model, a script is included in the notebook to generate a json with the appropriate format. In the following path, the annotations for the three video sequences used have been uploaded, ideal for carrying out tests on the models obtained after Fine-Tuning.

* https://github.com/IvanGarcia7/NGSIM-Dataset-Annotations


# REQUIREMENTS:

* Tensorflow 2
* Tensorflow Object Detection
* OpenCV
* Numpy


© 2021 Iván García Aguilar 
All Rights Reserved.
