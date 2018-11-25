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

# Main Image Segmentation Run Function (called in run.py main())
def run():
'''
This paper seemed promising for combining SIFT and Graph Cuts: http://www.me.cs.scitec.kobe-u.ac.jp/~takigu/pdf/2008/TuAT10.21.pdf
Executes image segmentation of likely food objects, return image segments as a set for each processed image.
Input: Preprocessed Image Set
Output: Image Segments as individual images in Segment folder
'''



if(__name__=="__main__"):
    run()