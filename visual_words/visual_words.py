import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from time import time

class visual_words:    
    def __init__(self):
        pass
        
def plot_images(n_rows, n_cols, images):    
    
    f = plt.figure()
    for i, image in enumerate(images):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(images[i], cmap = plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    #f.savefig('./figures/knn_faces.png')                
            

np.random.seed(0)

if __name__ == "__main__":    
    
    #Overview:
    #Olivetti dataset
    #Split into test and training
    #extract keypoints and compute sift features on training images        
    #cluster sift features into a visual dictionary of size V
    #represent each image as visual words histogram
    #apply tf-idf (need text data)    
    #fit LDA topic model on bags of visual words
    #given test data transform test image into tf_idf vector
    #use cosine similarity for image retrieval
    #display top-K images
                                                             
    # Load the faces datasets
    data = fetch_olivetti_faces(shuffle=True, random_state=0)
    targets = data.target
    
    data = data.images.reshape((len(data.images), -1))
    data_train = data[targets < 30]
    data_test = data[targets >= 30]
    num_train_images = data_train.shape[0]
        
    #show mean training image        
    plt.figure()
    plt.imshow(np.mean(data_train,axis=0).reshape(64,64))    
    plt.title('Olivetti Dataset (Mean Training Image)')    
    plt.show()
    
    #show random selection of images
    rnd_idx = np.arange(num_train_images)
    np.random.shuffle(rnd_idx)
    images = data_train[rnd_idx[0:16],:].reshape(16,64,64)
    plot_images(4,4,images)    
    
    #compute dense SIFT    
    num_kps = np.zeros(num_train_images)
    sift = cv2.SIFT()
    #orb = cv2.ORB()
    for img_idx in range(num_train_images):
        gray_img = 255*data_train[img_idx,:]/np.max(data_train[img_idx,:]) #scale
        gray_img = gray_img.reshape(64,64).astype(np.uint8)    #reshape and cast
    
        dense = cv2.FeatureDetector_create("Dense")
        kp = dense.detect(gray_img)
        kp, des = sift.compute(gray_img, kp)
        #kp, des = orb.compute(gray_img, kp)    
        #img_kp = cv2.drawKeypoints(gray_img, kp, color=(0,255,0), flags=0)
        #cv2.imshow('ORB keypoints', img_kp)
        
        num_kps[img_idx] = len(kp)
        #stack descriptors for all training images
        if (img_idx == 0):
            des_tot = des
        else:
            des_tot = np.vstack((des_tot, des))            
    #end for
    
    #cluster images into a dictionary
    dictionary_size = 100
    kmeans = MiniBatchKMeans(n_clusters = dictionary_size, init = 'k-means++', batch_size = 5000, random_state = 0, verbose=0)
    tic = time()
    kmeans.fit(des_tot)
    toc = time()
    kmeans.get_params()
    print "K-means objective: %.2f" %kmeans.inertia_    
    print "elapsed time: %.4f sec" %(toc - tic)

    kmeans.cluster_centers_
    labels = kmeans.labels_    
    
    #PCA plot of kmeans_cluster centers
    pca = PCA(n_components=2)
    visual_words = pca.fit_transform(kmeans.cluster_centers_)

    plt.figure()
    plt.scatter(visual_words[:,0], visual_words[:,1], color='b', marker='o', lw = 2.0, label='Olivetti visual words')
    plt.title("Visual Words (PCA of cluster centers)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.legend()
    plt.show()                
    
    #histogram of labels for each image = term-document matrix
    A = np.zeros((dictionary_size,num_train_images))
    ii = 0
    jj = 0
    for img_idx in range(num_train_images):
        if img_idx == 0:
            A[:,img_idx], bins = np.histogram(labels[0:num_kps[img_idx]], bins=range(dictionary_size+1))
        else:
            ii = np.int(ii + num_kps[img_idx-1])
            jj = np.int(ii + num_kps[img_idx])
            A[:,img_idx], bins = np.histogram(labels[ii:jj] , bins=range(dictionary_size+1))             
        #print str(ii) + ':' + str(jj)
    #end for
    plt.figure()
    plt.spy(A.T, cmap = 'gray')   
    plt.gca().set_aspect('auto')
    plt.title('AP tf-idf corpus')
    plt.xlabel('dictionary')
    plt.ylabel('documents')    
    plt.show()    
    
    #fit LDA topic model based on tf-idf of term-document matrix
    num_features = dictionary_size
    num_topics = 8 #fixed for LDA
                 
    #fit LDA model
    print "Fitting LDA model..."
    lda_vb = LatentDirichletAllocation(n_topics = num_topics, max_iter=10, learning_method='online', batch_size = 512, random_state=0, n_jobs=1)

    tic = time()
    lda_vb.fit(A.T)  #online VB
    toc = time()
    print "elapsed time: %.4f sec" %(toc - tic)
    print "LDA params"
    print lda_vb.get_params()

    print "number of EM iter: %d" % lda_vb.n_batch_iter_
    print "number of dataset sweeps: %d" % lda_vb.n_iter_

    #topic matrix W: K x V
    #components[i,j]: topic i, word j
    #note: here topics correspond to label clusters
    topics = lda_vb.components_
    
    f = plt.figure()
    plt.matshow(topics, cmap = 'gray')   
    plt.gca().set_aspect('auto')
    plt.title('learned topic matrix')
    plt.ylabel('topics')
    plt.xlabel('dictionary')
    plt.show()
    f.savefig('./figures/topic.png')
     
    #topic proportions matrix: D x K
    #note: np.sum(H, axis=1) is not 1
    H = lda_vb.transform(A.T)
    
    f = plt.figure()
    plt.matshow(H, cmap = 'gray')   
    plt.gca().set_aspect('auto')
    plt.show()
    plt.title('topic proportions')
    plt.xlabel('topics')
    plt.ylabel('documents')
    f.savefig('./figures/proportions.png')
        
    #given test data transform test image into tf_idf vector    
    #show mean test image        
    plt.figure()
    plt.imshow(np.mean(data_test,axis=0).reshape(64,64))    
    plt.show()
    
    num_test_images = data_test.shape[0]
    num_test_kps = np.zeros(num_test_images)
    #compute dense SIFT
    sift = cv2.SIFT()
    #orb = cv2.ORB()
    for img_idx in range(num_test_images):
        gray_img = 255*data_test[img_idx,:]/np.max(data_test[img_idx,:]) #scale
        gray_img = gray_img.reshape(64,64).astype(np.uint8)    #reshape and cast
    
        dense = cv2.FeatureDetector_create("Dense")
        kp = dense.detect(gray_img)
        kp, des = sift.compute(gray_img, kp)
        #kp, des = orb.compute(gray_img, kp)    
        #img_kp = cv2.drawKeypoints(gray_img, kp, color=(0,255,0), flags=0)
        #cv2.imshow('ORB keypoints', img_kp)
        
        num_test_kps[img_idx] = len(kp)
        #stack descriptors for all test images
        if (img_idx == 0):
            des_test_tot = des
        else:
            des_test_tot = np.vstack((des_test_tot, des))            
    #end for

    #assign des_test_tot to one of kmeans cluster centers    
    #use 128-dimensional kd-tree to search for nearest neighbors    
    kdt = KDTree(kmeans.cluster_centers_)
    Q = des_test_tot  #query
    kdt_dist, kdt_idx = kdt.query(Q,k=1) #knn
    test_labels = kdt_idx #knn = 1 labels
    
    #form A_test matrix from test_labels
    #histogram of labels for each image: term-document matrix
    A_test = np.zeros((dictionary_size,num_test_images))
    ii = 0
    jj = 0
    for img_idx in range(num_test_images):
        if img_idx == 0:
            A_test[:,img_idx], bins = np.histogram(test_labels[0:num_kps[img_idx]], bins=range(dictionary_size+1))
        else:
            ii = np.int(ii + num_kps[img_idx-1])
            jj = np.int(ii + num_kps[img_idx])
            A_test[:,img_idx], bins = np.histogram(test_labels[ii:jj] , bins=range(dictionary_size+1))             
        #print str(ii) + ':' + str(jj)
    #end for
    plt.figure()
    plt.spy(A_test.T, cmap = 'gray')   
    plt.gca().set_aspect('auto')
    plt.title('AP tf-idf corpus')
    plt.xlabel('dictionary')
    plt.ylabel('documents')
    plt.show()                 
    
    #Use fit transform on A_test for already trained LDA to get the H_test matrix
    #topic proportions matrix: D x K
    #note: np.sum(H, axis=1) is not 1
    H_test = lda_vb.transform(A_test.T)    
    f = plt.figure()
    plt.matshow(H_test, cmap = 'gray')   
    plt.gca().set_aspect('auto')
    plt.show()
    plt.title('topic proportions')
    plt.xlabel('topics')
    plt.ylabel('documents')
    f.savefig('./figures/proportions_test.png')
    
    #retrieve H_train document that's closest in cosine similarity for each H_test
    #use cosine similarity for image retrieval
    Kxy = cosine_similarity(H_test, H)
    knn_test = np.argmin(Kxy, axis=1)      
    f = plt.figure()
    plt.matshow(Kxy, cmap = 'gray')   
    plt.gca().set_aspect('auto')
    plt.show()
    plt.title('Cosine Similarity')
    plt.xlabel('train data')
    plt.ylabel('test data')
    f.savefig('./figures/cosine_similarity.png')                    

    #display knn images (docId is an image)
    rnd_idx = np.arange(num_test_images)
    np.random.shuffle(rnd_idx)
    images = data_test[rnd_idx[0:16],:].reshape(16,64,64)
    images_knn = data_train[knn_test[rnd_idx[0:16]],:].reshape(16,64,64)    
    plot_images(4,4,images)
    plot_images(4,4,images_knn)
                                                                