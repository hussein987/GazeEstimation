# Eye Gaze Tracking

In this project, we shall create a system for eye-gaze tracking. The system uses deep learning and image recognition to predict the eye-gaze direction. The project is inspired by the following paper [L2CS-NET: FINE-GRAINED GAZE ESTIMATION IN UNCONSTRAINED ENVIRONMENTS](https://arxiv.org/pdf/2203.03339v1.pdf)







## Overview
Eye tracking is becoming a very important capability across many domains, including security, psychology, computer vision, and medical diagnosis. Also, gaze is important for security applications to analyze suspicious gaze behavior. A use case in educational institutes is the automated analysis of the student’s eye gazes during an examination to help minimize malpractices.
In this project, we're going to implement a CNN-based appearance-based deep learning solution for eye gaze tracking on images and video streams, and ultimately deploying the solution in real-time settings.

The main novelty of [L2CS-NET](https://arxiv.org/pdf/2203.03339v1.pdf) compared to other models is summarized in two main ideas:
* Predict each gaze angle (yaw and pitch) separately using two fully-connected layers instead of one. These two fully-connected layers share the same convolution layers (ImageNet-pretrained ResNet-50) in the backbone.

* They use a combined loss function, that doesn’t only consider the mean squared error (l2), but also adds a cross entropy loss for classification, this is done by splitting up the continuous gaze target in each dataset (pitch and yaw angles) into bins with binary labels for classification. The final loss function is a linear combination between the mse and cross entropy loss as follows:
        $$CLS(y,p)=CrossEntropy(y,p)+MSE(y,p)$$
According to the literature, none of the current state-of-the-art models uses such a combined loss function.

However, it suffer from a bad performance in real-time
scenarios as it solely focuses on achieving good accuracy
score on the used dataset while discarding the inference time
analysis and testing. I have tested this approach on real-time
environment (predicting over a stream of frames), and I got
a frame-rate of 1.8 FPS, this test has been done using the
M1 Apple’s Metal Performance Shaders (MPS) as a backend
for PyTorch, and the results are showed in the attached git
repository. The following gif shows how slow a video should be for the L2C to work well.
![Alt text](./images/gaze.gif?raw=true)

To enhance the training speed and adapt the system for
real-time settings, the idea is to replace the backbone net-
work (pretrained ResNet 50) shown in Fig.3 by an [efficientNet](https://arxiv.org/pdf/1905.11946.pdf) to decrease the number of parameters by a factor X and
increase the training/inference speed by a factor Y. By doing
that, the prediction time will be decreased, and therefore an
enhanced ability to provide an up-to-date frame-by-frame
prediction will be gained. This is not the first



## Dataset

### Datasets for the proposed architecture
To train and evaluate my models, we use a dataset collected with unconstrained settings: MPIIGaze.
* [MPIIGaze](http://gaze360.csail.mit.edu/) provides 213.659 images from 15 subjects
captured during their daily routine over several months. Consequently, it contains images with diverse backgrounds, time,
and lighting that make it suitable for unconstrained gaze estimation. It was collected using software that asks the participants to look at randomly moving dots on their laptops.

### Dataset used to test EfficientNet (as a proof of concept)
For now, I'm using [UnityEye](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) dataset (for generating synthetic images of eye regions along with their labels i.e. the direction of where the eye is looking) to test the implementation of the efficientNet before encorporating it into the proposed architecture. This data generation method combines a novel generative 3D model of the human eye region with a real-time rendering framework. The model is based on high-resolution 3D face scans and uses real-time approximations for complex eyeball materials and structures as well as anatomically inspired procedural geometry methods for eyelid animation. The training set of UnityEyes captures a large degree of appearance variation, which enables us to test against challenging images.
![Alt text](./images/unityeye.png?raw=true "synthetic data using generative 3D eye region model" )


## Results

Here are the results of real-time testing of L2C,
https://user-images.githubusercontent.com/49820108/206873051-423b1781-4d11-4fc1-9212-4e885c3fd7c9.mp4


Here are the results of the real-time testing of L2C with EfficientNet
https://user-images.githubusercontent.com/49820108/206873167-dfa9bcb0-e792-4400-9757-02e8e2b91693.mov


## How to run

* Set up a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```
* Install required packages:
```
pip install -r requirements.txt  
```

* Install the face detector:
```sh
pip install git+https://github.com/elliottzheng/face-detection.git@master
```
*  Run:
```
 python demo.py \
 --snapshot models/L2CSNet_gaze360.pkl \
 --gpu 0 \
 --cam 0 \
```
