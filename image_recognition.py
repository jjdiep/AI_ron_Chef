#!/usr/bin/env python
'''
File running image recognition for Food Image Recognition and Search
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

# useful reference http://www.cs.cornell.edu/~ylongqi/paper/YangYumme.pdf
# tensorflow library example, 
def train():
# run on training set

def test():
# run on sample image, establish baseline on test images

def init():
# etc.

# Main Image Segmentation Run Function (called in run.py main())
def run():
'''
Input: Segmented Image Set
Output: Feature Descriptors
'''



if(__name__=="__main__"):
    run()