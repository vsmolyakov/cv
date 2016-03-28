from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

from skimage import data
from skimage.util import img_as_float
from skimage.feature import (corner_harris, corner_subpix, corner_peaks, plot_matches)
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac

def gaussian_weights(window_ext, sigma=1):
    y, x = np.mgrid[-window_ext:window_ext+1, -window_ext:window_ext+1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
    g /= 2 * np.pi * sigma * sigma #normalize
    return g
    
#match corners based on min Gaussian weighed SSDs distance
#between key-point patches
def match_corner(coord, window_ext=5):
    r, c =  np.round(coord).astype(np.intp)
    window_orig = img_orig[r-window_ext:r+window_ext+1,
                           c-window_ext:c+window_ext+1, :]

    # weight pixels depending on distance to center pixel
    weights = gaussian_weights(window_ext, 3)
    weights = np.dstack((weights, weights, weights))

    # compute sum of squared differences to all corners in warped image
    SSDs = []
    for cr, cc in coords_warped:
        window_warped = img_warped[cr-window_ext:cr+window_ext+1,
                                   cc-window_ext:cc+window_ext+1, :]
        SSD = np.sum(weights * (window_orig - window_warped)**2)
        SSDs.append(SSD)

    # use corner with minimum SSD as correspondence
    min_idx = np.argmin(SSDs)
    #import pdb; pdb.set_trace()
    
    return coords_warped_subpix[min_idx]    


if __name__ == "__main__":

    #checkerboard image
    checkerboard = img_as_float(data.checkerboard())
    img_orig = np.zeros(list(checkerboard.shape) + [3])
    img_orig[...,0] = checkerboard #equivalent to img_orig[:,:,0]
    
    gradient_r, gradient_c = np.mgrid[0:img_orig.shape[0], 0:img_orig.shape[1]] / float(img_orig.shape[0])
    img_orig[...,1] = gradient_r
    img_orig[...,2] = gradient_c
    
    img_orig = rescale_intensity(img_orig)
    img_orig_gray = rgb2gray(img_orig)
    
    plt.imshow(img_orig_gray,cmap=plt.cm.gray)
    plt.show()
    
    #warp synthetic image
    tform = AffineTransform(scale=(0.9,0.9), rotation=0.2, translation=(20,-10))
    img_warped = warp(img_orig, tform.inverse, output_shape=(200,200))
    img_warped_gray = rgb2gray(img_warped)

    plt.imshow(img_warped_gray,cmap=plt.cm.gray)
    plt.show()

    # extract corners using Harris' corner measure
    coords_orig = corner_peaks(corner_harris(img_orig_gray), threshold_rel=0.001,
                           min_distance=5)
    coords_warped = corner_peaks(corner_harris(img_warped_gray),
                             threshold_rel=0.001, min_distance=5)

    # determine sub-pixel corner position
    coords_orig_subpix = corner_subpix(img_orig_gray, coords_orig, window_size=9)
    coords_warped_subpix = corner_subpix(img_warped_gray, coords_warped,window_size=9)
            
    # find correspondences using simple weighted sum of squared differences
    src = []
    dst = []
    for coord in coords_orig_subpix:
        src.append(coord)
        dst.append(match_corner(coord))
    src = np.array(src)
    dst = np.array(dst)
        
    # estimate affine transform model using all coordinates
    model = AffineTransform()
    model.estimate(src, dst)    
    
    # robustly estimate affine transform model with RANSAC
    model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3, residual_threshold=2, max_trials=100)
    outliers = (inliers == False)

    # compare true, estimated and robust transform parameters
    print(tform.scale, tform.translation, tform.rotation)
    print(model.scale, model.translation, model.rotation)
    print(model_robust.scale, model_robust.translation, model_robust.rotation)                
    
    # visualize correspondence using plot_matches
    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray() #gray scale colormap

    inlier_idxs = np.nonzero(inliers)[0]
    plot_matches(ax[0], img_orig_gray, img_warped_gray, src, dst,
             np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
    ax[0].axis('off')
    ax[0].set_title('Correct correspondences')

    outlier_idxs = np.nonzero(outliers)[0]
    plot_matches(ax[1], img_orig_gray, img_warped_gray, src, dst,
             np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
    ax[1].axis('off')
    ax[1].set_title('Faulty correspondences')

    plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    