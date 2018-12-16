import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage import color
from skimage.measure import regionprops
# import eta.core.image as etai
from collections import deque
import pdb
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import SpectralClustering
from matplotlib import patches
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import itertools
from scipy import linalg
from sklearn import mixture

def getSIFTfeatures(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    kp=sift.detect(gray,None)
    img1=cv2.drawKeypoints(gray,kp,None)
    pts=np.asarray([[p.pt[0], p.pt[1]] for p in kp])
    cols=pts[:,0]
    rows=pts[:,1]
    return pts

def clusterSIFTfeatures(img):

    #Gaussian Mixture Model
    X=getSIFTfeatures(img)
    lowest_bic=np.infty
    bic=[]
    n_components_range=range(1,4)
    cv_types=['spherical','tied','diag','full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm=mixture.GaussianMixture(n_components=n_components,covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic=bic[-1]
                best_gmm=gmm
    bic=np.array(bic)
    color_iter=itertools.cycle(['navy','turquoise','cornflowerblue','darkorange'])
    clf=best_gmm
    bars=[]

    # All SIFT Features
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    kp=sift.detect(gray,None)
    img1=cv2.drawKeypoints(gray,kp,None)

    # Plot
    fig,ax=plt.subplots(1)
    ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    
    # GMM Overlay
    ell_width=[]
    ell_height=[]
    ell_center=[]
    Y_=clf.predict(X)
    for i, (mean,cov,color) in enumerate(zip(clf.means_,clf.covariances_,color_iter)):
        v,w=linalg.eigh(cov)
        if not np.any(Y_==i):
            continue
        angle=np.arctan2(w[0][1],w[0][0])
        angle=180.*angle/np.pi
        v=2.*np.sqrt(2.)*np.sqrt(v)
        ell=mpl.patches.Ellipse(mean,v[0],v[1],180.+angle,color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(.5)
        ax.add_patch(ell)
        ell_width.append(v[0])
        ell_height.append(v[1])
        ell_center.append(mean)

    # Plot Gaussian Subpopulations
    plt.show()

    # Store BBox Information for Subimage Extraction
    ell_width=np.array(ell_width,dtype=np.uint64)
    ell_height=np.array(ell_height,dtype=np.uint64)
    BoundingBoxCenter=np.array(ell_center,dtype=np.uint64)
    BoundingBoxLength=ell_width

    for i in range(ell_width.shape[0]):
        if ell_height[i] > BoundingBoxLength[i]:
            BoundingBoxLength[i]=ell_height[i]

    return BoundingBoxCenter,BoundingBoxLength

def GaussianSubImage(img,BoundingBoxCenter,BoundingBoxLength):

    """#Plot Original Image
    cv2.imshow('test',img)
    esc=cv2.waitKey(0)
    if esc==20:
        cv2.destroyAllWindows()
    """

    subimages=[]
    crop=np.amax(BoundingBoxLength)
    img=cv2.copyMakeBorder(img,crop,crop,crop,crop,cv2.BORDER_CONSTANT,value=[0,0,0])

    """#Plot Padded Image
    cv2.imshow('test',img)
    esc=cv2.waitKey(0)
    if esc==20:
        cv2.destroyAllWindows()
    """

    for i in range(BoundingBoxLength.shape[0]):
        x,y=BoundingBoxCenter[i]
        length=BoundingBoxLength[i]
        length=np.uint64(length*0.7)
        x=np.uint64(x+crop)
        y=np.uint64(y+crop)
        top=np.uint64(y+length)
        bottom=np.uint64(y-length)
        right=np.uint64(x+length)
        left=np.uint64(x-length)
        sub=img[bottom:top,left:right]
        subimages.append(sub)
    subimg=np.array(subimages)

    #Plot Subimages
    for i in range(subimg.shape[0]):
	    cv2.imshow('test',subimg[i])
	    esc=cv2.waitKey(0)
	    if esc==20:
	        cv2.destroyAllWindows()

    return subimg

def dir_maker():
    # evidently Linux and Windows dependent but...
    path = 'data/new_testimage/apple'
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, path)

    if not os.path.exists(final_directory):
        try:
            os.makedirs(final_directory)
        except OSError:
            print("Creation of the directory failed" % path)
        else:
            print("Successfully created the directory %s" % path)

    return final_directory

def main():
    '''
    Once initial cluster image opens, close window and use keyboard commands to step through subimages.
    '''
    img=cv2.imread('testsample_1.png')
    x_dim, y_dim, c_dim = img.shape
    min_len = 1100
    max_dim = max(x_dim,y_dim)
    if max_dim < min_len:
        max_dim = min_len
    img = cv2.resize(img, (0,0), fx = min_len/max_dim, fy = min_len/max_dim)
    BoundingBoxCenter,BoundingBoxLength=clusterSIFTfeatures(img)
    sub=GaussianSubImage(img,BoundingBoxCenter,BoundingBoxLength)
    dir_maker()
    sub_min_len = 500

    for i in range(len(sub)):
        sub_x_dim, sub_y_dim, sub_c_dim = sub[i].shape
        sub_max_dim = max(sub_x_dim, sub_y_dim)
        delta_w = y_dim - sub_y_dim
        delta_h = x_dim - sub_x_dim
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0]
        # sub_img = cv2.resize(sub[i], (0,0), fx = 1, fy = 1)
        sub_img = cv2.resize(sub[i], (0,0), fx = sub_min_len/sub_max_dim, fy = sub_min_len/sub_max_dim)
        # sub_img = cv2.copyMakeBorder(sub[i], top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        cv2.imwrite('data/new_testimage/apple/subimage_'+str(i)+'.jpg',sub_img)


if(__name__=="__main__"):
    main()
