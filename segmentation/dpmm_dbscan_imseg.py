
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_sample_images
from sklearn.preprocessing import StandardScaler

from sklearn import mixture
from sklearn.cluster import DBSCAN

from time import time

np.random.seed(0)

if __name__ == "__main__":
    
    #dataset = load_sample_images()    
    #img1 = dataset.images[0]

    img = cv2.imread('./figures/cyclists.png')
    num_rows, num_cols = img.shape[:2]
    cv2.imshow('original', img)
    
    #convert to [L a b x y]
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cv2.imshow('LAB space', lab_img)
    
    I_lab = np.reshape(lab_img, (num_rows*num_cols, 3))
    
    xx = np.linspace(0, num_cols-1, num_cols)
    yy = np.linspace(0, num_rows-1, num_rows)        
    xv, yv = np.meshgrid(xx, yy)
    
    I_x = xv.reshape(num_rows*num_cols,1)
    I_y = yv.reshape(num_rows*num_cols,1)    
    
    #compose feature vectors:
    Y = np.hstack((I_lab, I_x, I_y))
    Ysc = StandardScaler().fit_transform(Y)
    
    #scale each dimension
    W = np.array([3, 3, 3, 1, 1])
    W = 10*W/float(np.sum(W))
    
    #component-wise multiply
    Ysc = Ysc * W
    
    #On-line VB for DPGMM
    print "running DPGMM..."
    dpgmm = mixture.DPGMM(n_components = 4, covariance_type = 'full', alpha = 10, n_iter = 10, tol = 1e-4, verbose=1)    
    tic = time()
    dpgmm_labels = dpgmm.fit_predict(Ysc)    
    toc = time()    
    dpgmm_posterior = dpgmm.predict_proba(Ysc)
    dpgmm.get_params()
    print "elapsed time: %.4f sec" %(toc - tic)
    
    I_label = np.reshape(dpgmm_labels, (num_rows, num_cols, 1))
    I_seg = I_label * 255.0 / np.max(I_label)  #scale
    I_seg = I_seg.astype(np.uint8)             #cast
    I_seg_bgr = cv2.cvtColor(I_seg, cv2.COLOR_GRAY2BGR)
    cv2.imshow('DPGMM seg', I_seg_bgr)
    #cv2.imwrite('./figures/dpgmm_seg.png', I_seg_bgr)
        
    #DB-SCAN
    print "running DBSCAN..."
    dbscan = DBSCAN(eps=0.5, min_samples=50, metric='euclidean', algorithm='auto')
    tic = time()
    dbscan_labels = dbscan.fit_predict(Ysc)
    toc = time()
    print "elapsed time: %.4f sec" %(toc - tic)
    dbscan.get_params()
        
    #noise samples are labeled -1
    dbscan_labels = dbscan_labels + 1
    
    I2_label = np.reshape(dbscan_labels, (num_rows, num_cols, 1))
    I2_seg = I2_label * 255.0 / np.max(I2_label)  #scale
    I2_seg = I2_seg.astype(np.uint8)              #cast
    I2_seg_bgr = cv2.cvtColor(I2_seg, cv2.COLOR_GRAY2BGR)
    cv2.imshow('DPGMM seg', I2_seg_bgr)
    #cv2.imwrite('./figures/dbscan_seg.png', I2_seg_bgr)