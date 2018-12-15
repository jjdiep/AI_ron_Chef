import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage import color
from skimage.measure import regionprops
import eta.core.image as etai
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
    n_components_range=range(1,7)
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
        length=np.uint64(length*0.6)
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


def main():
    # LOAD IMAGE
    #img = etai.read('porch1.png')[:,:,:3]
    #img=cv2.imread('porch1.png')
    #img=cv2.imread('Fridge.PNG')
    img=cv2.imread('LotsofFood.PNG')
    #img=cv2.imread('ManyFruits.PNG')
    #img=cv2.imread('Banana.PNG')
    #img=cv2.imread('breakfastburrito.PNG')
    #img=cv2.imread('bruschetta.PNG')
    #img=cv2.imread('lettuceEASY.PNG')
    #img=cv2.imread('lettuceHARD.PNG')
    #img=cv2.imread('lettuceMEDIUM.PNG')
    #img=cv2.imread('tunatartar.PNG')
    #img=cv2.imread('waffle.PNG')
    #img=cv2.imread('JustinsVegggies.PNG')
    #img=cv2.imread('JasonsFridge.JPG')
    #img=cv2.imread('WadesFridge.JPG')
    #img=cv2.imread('FelmansFridge.JPG')
    #img=cv2.imread('DrewsFridge.JPG')
    #img=cv2.imread('ChrissFridge.JPG')
    #img=cv2.imread('IMG_0925.JPG')

    BoundingBoxCenter,BoundingBoxLength=clusterSIFTfeatures(img)
    sub=GaussianSubImage(img,BoundingBoxCenter,BoundingBoxLength)



if(__name__=="__main__"):
    main()
