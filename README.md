# Self-Supervised Homography Estimation
## University of Alberta
## CMPUT 615 - Final Project
## April 2022

### Abstract
Homography matrix plays a key role in multiview geometry and computer vision and has variousapplications such as image alignment, image rectification, and image registration. The classic methodsfor estimating the homography matrix between two views of a same object requires a prior knowledge on point correspondences as well as constraints such as brightness constancy. In this project, we propose a self-supervised/unsupervised deep learning method which leverages the power of high dimensional features to levitate the requirement for point correspondence and brightness constancy. We evaluate the proposed method on the Brick Motion data of the UCSB dataset and consider four different types of motions, namely panning, rotation, perspective transformation, and scaling, and we show that the proposed method demonstrates a reasonably good performance.

### Package Requirements and Dependencies
This project is written in **Python 3.10** and uses the following packages
* torch 1.11 + cu 11.3
* torchgeometry 0.1.2
* opencv 4.5.5
* scikit-image 0.19.2

### Implementation Details
* All files are included in the *source* directory
* The four pairs of images tested for this project are in *source/datasets* directory. Images are paired as *a* and *b*.
* The neural layers and the whole model can be found in *source/networks.py*.
* The training procedure is available at *source/train.py*.
* all utility functions can be accessed from *source/utils.py*.