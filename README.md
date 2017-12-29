# cv
computer vision

### Description

**Manifold Learning**

Images are high dimensional objects that live on manifolds. The figure below shows t-SNE embedding of 64 dimensional digits dataset in 2-D. A KD-tree is used to find nearest neighbors  to an image query on the manifold.

<p align="center">
<img src="https://github.com/vsmolyakov/cv/blob/master/manifold/figures/manifold_merged.png"/>
</p>

References:  
*L. Van Der Maaten, "Visualizing Data using t-SNE", JMLR 2008*  
*R. Szeliski, "Computer Vision: Algorithms and Applications", 2010.*  

**Visual Words**

Image search can be formulated as topic similarity for a topic model build from visual words. Visual words are cluster centers of image features extracted at image keypoints. In this example, dense SIFT features are extracted from a collection of 10 face images of 40 different people (the Olivetti faces dataset). The 128-dimensional SIFT features are clustered using mini-batch K-means into a dictionary of K visual words. An online variational bayes algorithm is used to learn the LDA topic model and extract topic proportions for training image data. Test image data is converted into topic space and training images are retrieved based on cosine similarity between topic proportions of the train and test images.

<p align="center">
<img src="https://github.com/vsmolyakov/cv/blob/master/visual_words/figures/visual_words_merged.png"/>
</p>

The figure above shows the Olivetti dataset mean training image (left) followed by the PCA visualization of visual words (middle). The learned visual topics are shown at the bottom. A random sample of test images along with the retrieved nearest neighbors are shown on the right.

Note that there is no overlap between training and test images (i.e. the individuals are different). Based on the retrieved nearest neighbors, we can see that certain faces explain the test images best, which suggests that ranking or post-processing of the retrieved results can improve performance.

References:  
*R. Szeliski, "Computer Vision: Algorithms and Applications", 2010.*  

**Segmentation**

Image segmentation is a very common problem in computer vision. We often like to identify objects in an image before further processing.

<p align="center">
<img src="https://github.com/vsmolyakov/cv/blob/master/segmentation/figures/seg_merged.png"/>
</p>

The figure above compares two non-parametric algorithms: DP-GMM and DBSCAN applied to image segmentation. From the images we can see that different parameter settings resulted in different discovery of clusters or segments in an image. The on-line VB algorithm configured with alpha = 10 produced a binary segmentation separating the road from the rest of the image. In contrast, DBSCAN based on kd-tree algorithm produced 15 clusters. Both algorithms do not assume that the number of segments in an image is known ahead of time.

References:  
*D. Blei and M. Jordan, "Variational Inference for Dirichlet Process Mixtures", Bayesian Analysis, 2006*

*Ester, M., H. P. Kriegel, J. Sander, and X. Xu, “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise”, AAAI, 1996*

**Conv Nets**

Convolutional Neural Networks (CNN) revolutionized the field of computer vision. Trained on millions of images, CNNs such as AlexNet perform very well on a variety of vision tasks.

<p align="center">
<img src="https://github.com/vsmolyakov/cv/blob/master/caffe/figures/alexnet_merged.png"/>
</p>

The figure above depicts AlexNet classifier trained on ImageNet used to classify a super-pixelized image of an anemone fish using the fastSCSP algorithm. Interestingly enough, with K=286 superpixels, the AlexNet correctly predicts the two ground truth categories: anemone fish and sea anemone as indicated by the two spikes in the right most figure.

References:  
*O. Friefeld, Y. Li, and J. W. Fisher III, "A fast method for inferring high-quality simply-connected superpixels", ICIP 2015*  

**Image Search**

A VGG-16 convolutional neural net is used to construct an image representation in latent space based on a vector of activations.
Images from Caltech101 dataset are mapped into activation vectors and a query image is used to find K nearest distance neighbors.

<p align="center">
<img src="https://github.com/vsmolyakov/cv/blob/master/image_search/figures/image_search_merged.png"/>
</p>

The figure above shows the query image of a chair (left) and a retrieved nearest neighbor images (right). The retrieved images closely resemble the query image.

References:  
*K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", ICLR 2015*  

**Kalman Filter Tracking**

The Kalman filter is an algorithm for exact Bayesian filtering of Linear Dynamic System (LDS) models. The Kalman update step mu_n = Amu_{n-1} + K_n (x_n - CAmu_{n-1}) consists of the previous value and a correction term multiplied by the Kalman gain K_n. The Kalman filter outputs a marginal distribution p(z_n|x_1,...,x_n), while the Kalman smoother takes all observations into account.

<p align="center">
<img src="https://github.com/vsmolyakov/cv/blob/master/kalman/figures/kalman_merged.png"/>
</p>

The figure above shows the predictions of Kalman filter (left) and Kalman smoother (right). We can see that the Kalman smoother produces a smooth trajectory due both forward and backward update steps.

References:  
*C. Bishop, "Pattern Recognition and Machine Learning", 2006*  

**Particle Filter**

Particle filter is a Sequential Monte Carlo method of estimating the internal states of a Switching Linear Dynamic System (SLDS). A set of particles is used to represent the posterior distribution of the states of the Markov process. At each iteration, the particles get re-sampled and the ones that explain the observations best propagate to the next iteration.

<p align="center">
<img src="https://github.com/vsmolyakov/cv/blob/master/particles/figures/particle_filter_merged.png"/>
</p>

The figure above shows the generated SLDS states (left) and the inferred states (center) by Particle Filter (PF) and the Rao-Blackwellized version (RBPF). We can see that the inferred states closely correspond to the ground truth. Also shown is a particle re-sampling step (right) where only a fraction of particles survive to the next iteration.

References:  
*Nando de Freitas, "Rao-Blackwellized Particle Filter for Fault Diagnosis", 2002*  


**Siamese Neural Network**

Siamese Neural Network consists of two identical networks with shared weights tied together by a contrastive loss function. The network is trained to find similarities between inputs that can take on different modalities such as images (CNNs) or sentences (RNNs).

<p align="center">
<img src="https://github.com/vsmolyakov/cv/blob/master/siamese/figures/siamese_merged.png"/>
</p>

The figure above shows a siamese network applied to pairs of MNIST digits. The training set alternates between positive (matching) and negative (different) pairs of digits. The network is able to achieve 99.5% accuracy after 20 training epochs. Besides image recognition, siamese architectures have been used in signature matching and sentence similarity.

References:  
*G. Koch, et. al., "Siamese Neural Networks for One-shot Image Recognition", ICML 2015*  

**Generative Adversarial Network**

Generative Adversarial Network (GAN) consists of two networks trained simultaneously: a generative model (G) that generates output example given random noise as input and a discrimantive model (D) that distinguishes between real and generated examples.

<p align="center">
<img src="https://github.com/vsmolyakov/cv/blob/master/gan/figures/dcgan_cifar10.png"/>
</p>

As a result of training, the generative model learns to produce output examples that look very realistic (see CIFAR10 example above) recovering distribution of training data; while the discriminative network outputs equal class probability for real vs generated data.

References:  
*I. Goodfellow, et. al., "Generative Adversarial Networks", 2014  
F. Chollet, "Deep Learning with Python", Manning Publications, 2017*  

**Neural Style Transfer**

Neural style transfer takes as input a content image and a style image and produces an artistic rendering of the content image. It does that by minimizing the sum of content loss and style loss. The content of the image is captured by the higher layers of a CNN, while the style of the image refers to its texture and simple geometric shapes represented in the lower layers of a CNN.

<p align="center">
<img src="https://github.com/vsmolyakov/cv/blob/master/style_transfer/figures/style_transfer_merged.png"/>
</p>

The figure above shows the content image (left), the style image (middle) and the output image (right). The output image is found by minimizing the content and style loss for a VGG19 CNN using L-BFGS solver. The output image is the result of 20 iterations of L-BFGS.

References:  
*L. Gatys, A. Ecker, and M. Bethge, "A Neural Algorithm of Artistic Style", arXiv 2015  
F. Chollet, "Deep Learning with Python", Manning Publications, 2017*  


**Captions Generator**

An image caption generator is a multi-input neural network that consists of an image branch (ResNet50 CNN) and a language branch (LSTM). The two branches are merged into a common RNN that predicts the next word given the image and the previous caption words.

<p align="center">
<img src="https://github.com/vsmolyakov/cv/blob/master/captions/figures/captions_merged.png"/>
</p>

The neural captions generator is trained end-to-end on Flicker8K dataset. The training and validation loss and accuracy are shown in the figure above along with the generated captions using beam-search. The neural caption generator achieves a BLEU score of 0.61 on the test dataset.

References:  
*O. Vinyals, et. al., "Show and Tell: A Neural Image Caption Generator", CVPR 2015*  


### Dependencies

Matlab 2014a  
Python 2.7  
TensorFlow 1.3.0  
OpenCV  
