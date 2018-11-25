#!/usr/bin/env python
'''
File running image segmention for Food Image Recognition and Search
'''
# Standard Libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pdb
from skimage.segmentation import slic, mark_boundaries
from skimage import color
from skimage.measure import regionprops
import eta.core.image as etai
from collections import deque

def get_superpixel():

def get_correspondences():
# from SIFT/ORB keypoints?

def seg_neighbor():
# etc.

# Main Image Preprocessing Run Function (called in run.py main())
def run():
'''
Input: Original Image Set
Output: Preprocessed Image Set
'''



if(__name__=="__main__"):
    run()