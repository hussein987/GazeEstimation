# Eye Gaze Tracking

In this project, we shall create a system for eye-gaze tracking. The system uses deep learning and image recognition to predict the eye-gaze direction. The project is inspired by the following paper [L2CS-NET: FINE-GRAINED GAZE ESTIMATION IN UNCONSTRAINED ENVIRONMENTS](https://arxiv.org/pdf/2203.03339v1.pdf)


## Overview
Eye tracking is becoming a very important capability across many domains, including security, psychology, computer vision, and medical diagnosis. Also, gaze is important for security applications to analyze suspicious gaze behavior. A use case in educational institutes is the automated analysis of the student’s eye gazes during an examination to help minimize malpractices.
In this project, we're going to implement a CNN-based appearance-based deep learning solution for eye gaze tracking on images and video streams, and ultimately deploying the solution in real-time settings.\
Accordingly, I'm planning to encorporate the following contributions to the L2CS-NET:
1. Enhance the performance, I am currently implementing an idea for further enhancing the current performance, by adding a multi-channel network that takes eye images, full-face images, and face grid information as inputs, as suggested in Appearance-Based Gaze Estimation Using Dilated-Convolutions. The idea is not just to use the face images, but also provide single eye images as inputs as illustrated in the following diagram:\
![Alt text](./images/multichannel.png?raw=true)
However, before combining the channels into the fully connected layer FC1, we need to split the output of the backbone convolutional block into two fully connected layers, one for each gaze angle (yew and pitch), as in L2CS-Net, and apply the linear combination of loss functions.

2. Enhance the training speed and adapt the system for real-time settings, the idea behind that is to replace the backbone network (pretrained ResNet 50) by an [efficientNet](https://arxiv.org/pdf/1905.11946.pdf) to decrease the number of parameters by a factor X and increase the training/inference speed by a factor Y. We can see how L2CS-Net performs in video settings in the GIF below, and how the video needs to be slowed down a bit for it to provide an up-to-date frame-by-frame prediction.
![Alt text](./images/gaze.gif?raw=true)

## Dataset

### Datasets for the proposed architecture
To train and evaluate my models, I'm going to use the two popular datasets collected with unconstrained settings: Gaze360 and MPIIGaze.
* [Gaze360](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild): provides the widest range of 3D gaze annotations with a range of 360 degrees. It contains 238 subjects of
different ages, genders, and ethnicity. Its images are captured
using a Ladybug multi-camera system in different indoor and
outdoor environmental settings like lighting conditions and
backgrounds.
* [MPIIGaze](http://gaze360.csail.mit.edu/) provides 213.659 images from 15 subjects
captured during their daily routine over several months. Consequently, it contains images with diverse backgrounds, time,
and lighting that make it suitable for unconstrained gaze estimation. It was collected using software that asks the participants to look at randomly moving dots on their laptops.

### Dataset used to test EfficientNet (as a proof of concept)
For now, I'm using [UnityEye](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) dataset (for generating synthetic images of eye regions along with their labels i.e. the direction of where the eye is looking) to test the implementation of the efficientNet before encorporating it into the proposed architecture. This data generation method combines a novel generative 3D model of the human eye region with a real-time rendering framework. The model is based on high-resolution 3D face scans and uses real-time approximations for complex eyeball materials and structures as well as anatomically inspired procedural geometry methods for eyelid animation. The training set of UnityEyes captures a large degree of appearance variation, which enables us to test against challenging images.
![Alt text](./images/unityeye.png?raw=true "synthetic data using generative 3D eye region model" )

## Model
The architecture is described in the document, we have two main objectives, one is to speed up the training (Using efficient-Net) and the other one is to enhance the performance. However, it's not straight forward to do both at the same time, that's why I'm going first to test speeding up the training/inference (because it's simpler to test) by just replacing the backbone ResNet-50 by an EfficientNet, we’re predicting the direction of the gaze
vector, predicting the coordinates on the output image. After that, we’re using OpenCV to get the eye pupil’s position
and other eye’s landmarks, then connecting it to the coordi-
nates of the gaze vector to draw the gaze vector.
The current architecture uses EfficientNet replacing the output layers with 2 fully connected layers to match the intended
output size (2 in our case, the 𝑥 and 𝑦 positions of the gaze
vector).

## Code

- You can find the training results of EfficientNet [here](./source/predict_direction.ipynb).

Other scripts are still under development.

## Next steps

1. Embed EfficientNet into L2CS-Net, train and test it using MPIIGaze and Gaze360.
2. Video predictions
3. Realtime settings
4. Implement the proposed modifications of L2CS-Net architecture to enhance the performance.