# Zero-Shot Object Detection

This code is the testing implementation of the following work:

Shafin Rahman, Salman Khan, and Fatih Porikli. 
"Zero-Shot Object Detection: Learning to Simultaneously Recognize and Localize Novel Concepts." 
arXiv preprint arXiv:1803.06049 (2018). ([Project Page](https://salman-h-khan.github.io/ProjectPages/ZSD_Arxiv18.html))

![ProblemFigure](https://salman-h-khan.github.io/images/Fig1_ZSD_P.JPG)

## Requirements
* Download the pre-trained model available on the link below and place it inside the "Model" directory
([Link to pre-trained model (h5 format)](https://www.dropbox.com/s/v6ueoa1g19bddao/model_frcnn.hdf5?dl=0)) 
* Other requirements:
	- Python 2.7 
	- Keras 2.1.4
	- OpenCV 2.4.13
	- Tensorflow 1.3.0

This code has also been tested with Python 3.6, Keras 2.0.8, OpenCV 3.4.0 and on Ubuntu and Windows.


## Files
`sample_input.txt`: a sample input file containing test image paths
`detect.py`: to perform zero-shot detection task using sample_input.txt
`keras_frcnn`: directory containing supporting code of the model
`Dataset`: directory containing sample input and output
`Model`: directory containing pre-trained model and configuration file
`ImageNet2017`
  - `cls_names.txt`: list of 200 class names of ImageNet detection dataset.
	- `seen.txt`: list of seen class names used in the paper
	- `unseen.txt`: list of unseen class names used in the paper
	- `train_seen_all.zip`: it is a zipped version of text file `train_seen_all.txt` which contain training image paths and annotation used in the paper. Each line contains training image path, a bounding box co-ordinate and the ground truth class name of that bounding box.
		For example, Filepath,x1,y1,x2,y2,class_name
	- `unseen_test.txt`: all the image paths used for testing in the papers. Images are from training and validation set from ImageNet 2017 detection challenge. Every image contains at least one instance of unseen object.
	- `word_w2v.txt`: word2vec word vectors of 200 classes + bg used in the paper. The ith column represents the 500-dimensional word vectors of the class name of ith row of cls_names.txt.
	- `word_glo.txt`: GloVe word vectors of 200 classes + background (bg) used in the paper. The ith column represents the 300-dimensional word vectors of the class name of ith row of `cls_names.txt`.
	

## Running instruction
To run zero-shot detection on sample input kept in `Dataset/Sampleinput`, simply run detect.py after installing all dependencies like Keras, Tensorflow, OpenCV and placing the pre-trained model in the `Model` directory. This code will generate the output files for each input image to `Dataset/Sampleoutput`.

## Notes on ImageNet experiments
The resources required to reproduce results of ImageNet related experiments are kept in the directory ImageNet2017. All the images are from `ILSVRC2017_DET.tar.gz` which can be obtained from ImageNet detection challenge 2017 website. For both training and testing of this paper, we have used images from `/ILSVRC/Data/DET/train` and `/ILSVRC/Data/DET/val` of the zipped arxiv `ILSVRC2017_DET.tar.gz`.


## Trubleshooting
* If you get the `CUDA_ERROR_OUT_OF_MEMORY` in Tensorflow, place the following snippet after library loadings (the top section) in `detect.py`
	```python
	import tensorflow as tf 
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.8 #Change it to suit your GPU load
	K.set_session(tf.Session(config=config)) 
	```
	
* If you want to run the sample code on CPU only, place the following snippet after library loadings (the top section) in `detect.py`
	```python
	import tensorflow as tf 
	num_cores = 2 # 2,4, or 8
	config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : 1, 'GPU' : 0})
	K.set_session(tf.Session(config=config))
	```
	
## Citation 
### Citation
If you use this code and model for your research, please consider citing:
```
@article{rahman2018zeroshot, 
title={Zero-Shot Object Detection: Learning to Simultaneously Recognize and Localize Novel Concepts}, 
author={Rahman, Shafin and Khan, Salman and Porikli, Fatih}, 
journal={Asian Conference on Computer Vision}, 
year={2018} 
}
```

## Acknowledgment
We thank Yann Henon for the following implementation of Faster-RCNN: [keras-frcnn](https://github.com/yhenon/keras-frcnn)
