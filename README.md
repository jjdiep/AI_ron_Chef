# AI_ron_Chef
EECS 504 Project

Project Members:
Joseph Abrash
Justin Diep
Wayne Lao
Likai Sheng

Specific Dependencies Required:
IMPORTANT: pip install opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10 as SIFT is patented
keras
tensorflow
bs4
skimage, sklearn, scipy
h5py
Including others dependent on existing dependencies.

To run the (AI)ron_Chef algorithm:
0a) If needed, regenerate images with image_prep_test.py
0b) if model is untrained, run CNN.py to train weights
(NOTE: requires significant GPU computation power that may exceed most personal computers)

1) Run GMM_Segmentation.py to generate subimages
2) Run CNN_Test.py
3) Run recipe_crawler.py

Plot of CNN classification overlaying segmented image can be generated running CNN_testimageplot.py, however, it is not part of the code pipeline.

See output recipes in Recipe directory
