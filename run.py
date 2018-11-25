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
import image_search as img_search  # ok this is a bad name, pls change it to something that makes sense if you want to :)

# main executing function
def main():
'''
Executes image segmentation of likely food objects, object recognition
off of trained CNN, and runs recipe search using ingredient feature descriptors
identified. Returns food results and recipe. 
'''
prep_train_set = img_prep.run(orig_train_set)  # load images, misc image processing, both for actual image and training?
prep_user_img = img_prep.run(user_image)  # load images, misc image processing, both for actual image and training?
seg_train_set = img_seg.run(prep_image_set)
seg_user_img = img_seg.run(prep_user_img)
img_id.train(seg_train_set)  # only on initial training, will return trained weights for NN in a file
features = img_id.run(seg_user_img)  # should incorporate user feedback/identification for correction
search_output = img_search.run(features)  # tuneable to user preferences (maybe limit to say, cuisine, etc.)


if(__name__=="__main__"):
    main()