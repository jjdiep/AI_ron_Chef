#!/usr/bin/env python
'''
File running execution of Food Image Recognition and Search
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

# Defined functions
import image_preprocess as img_prep  # if necessary to preprocess multiple datasets...
import image_segmentation as img_seg
import image_recognition as img_id
import image_search as img_search  # ok this is a bad name, pls change it to something that makes sense :)

# main executing function
def main():
'''
Executes image segmentation of likely food objects, object recognition
off of trained CNN, and runs recipe search using ingredient feature descriptors
identified. Returns food results and recipe. 
'''
img_prep.run()
img_seg.run()
img_id.train()  # only on initial training
img_id.run()
img_search.run()


if(__name__=="__main__"):
    main()