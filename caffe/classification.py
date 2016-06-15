
import caffe
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import timeit
import pdb

import scipy.stats as stats

#global params
#caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
caffe_root = os.environ["CAFFE_ROOT"]
sys.path.insert(0, caffe_root + 'python')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#based on
#https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb


def vis_square(data, padsize=1, padval=0):
    
    # take an array of shape (n, height, width) or (n, height, width, channels)
    # and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)    
    
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    plt.show()
    

def compute_kl(p,q):
    
    p = p/np.sum(p)
    q = q/np.sum(q)
    
    return np.sum(p*np.log(p/q))


def main():

    #caffe.set_mode_cpu()
    caffe.set_device(0)
    caffe.set_mode_gpu()    
    
    net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # set net to batch size of 50
    net.blobs['data'].reshape(50,         #batch size
                               3,         #3-channel (BGR) image
                               227,227)   #image size is 277 x 277

    image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    
    #classify
    out = net.forward()    
    print "Predicted class is: ", out['prob'][0].argmax()
    #plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
    #plt.show()

    # load labels
    imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    except:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

    # sort top k predictions from softmax output (5 from the end in reverse order)
    output_prob = net.blobs['prob'].data[0].ravel()
    top_k = output_prob.argsort()[-1:-6:-1]
    print "top 5: "
    for idx in top_k:
        print str(output_prob[idx]) + ", " + str(labels[idx])

    #(batch_size, channel_dim, height, width)
    print "blob items:"
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape) 
    
    print "net params:"
    #(output_channels, input_channels, filter_height, filter_width)
    for layer_name, param in net.params.iteritems():
        print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)        

    #caffe.set_device(0)
    #caffe.set_mode_gpu()
    #net.forward()  # call once for allocation

    """
    # the parameters are a list of [weights, biases]
    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1))

    feat = net.blobs['conv1'].data[0, :36]
    vis_square(feat, padval=1)

    filters = net.params['conv2'][0].data
    vis_square(filters[:48].reshape(48**2, 5, 5))
    
    feat = net.blobs['conv2'].data[0, :36]
    vis_square(feat, padval=1)

    feat = net.blobs['conv3'].data[0]
    vis_square(feat, padval=0.5)

    feat = net.blobs['conv4'].data[0]
    vis_square(feat, padval=0.5)

    feat = net.blobs['conv5'].data[0]
    vis_square(feat, padval=0.5)
    
    feat = net.blobs['pool5'].data[0]
    vis_square(feat, padval=1)


    feat = net.blobs['fc6'].data[0]
    plt.subplot(2, 1, 1)
    plt.plot(feat.flat)
    plt.subplot(2, 1, 2)
    _ = plt.hist(feat.flat[feat.flat > 0], bins=100)


    feat = net.blobs['fc7'].data[0]
    plt.subplot(2, 1, 1)
    plt.plot(feat.flat)
    plt.subplot(2, 1, 2)
    _ = plt.hist(feat.flat[feat.flat > 0], bins=100)

    feat = net.blobs['prob'].data[0]
    plt.plot(feat.flat)

    # load labels
    imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    except:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    """
        
    #new image
    """
    img_sp_name = ['./figures/beaver.jpg',\
                   './figures/beaver_n_015_std_20_prior_count_20_num_sp_690_mean_hex.png',\
                   './figures/beaver_n_015_std_20_prior_count_20_num_sp_690_mean_hex_sample.png']
    """
    """
    img_sp_name = ['./figures/2.jpg',\
                   './figures/2_n_196_std_20_prior_count_20_num_sp_06_mean_hex.png',\
                   './figures/2_n_098_std_20_prior_count_20_num_sp_24_mean_hex.png',\
                   './figures/2_n_049_std_20_prior_count_20_num_sp_77_mean_hex.png',\
                   './figures/2_n_024_std_20_prior_count_20_num_sp_286_mean_hex.png',\
                   './figures/2_n_012_std_20_prior_count_20_num_sp_1100_mean_hex.png',\
                   './figures/uniform/2_n_196_std_20_prior_count_20_num_sp_06_mean_hex.png',\
                   './figures/uniform/2_n_098_std_20_prior_count_20_num_sp_24_mean_hex.png',\
                   './figures/uniform/2_n_049_std_20_prior_count_20_num_sp_77_mean_hex.png',\
                   './figures/uniform/2_n_024_std_20_prior_count_20_num_sp_286_mean_hex.png',\
                   './figures/uniform/2_n_012_std_20_prior_count_20_num_sp_1100_mean_hex.png']    
    """
    
    #TODO: loop over original images
    #      for each original get K_real number
    #      how to store KL? use list of lists
    IMG_PATH = '/afs/csail.mit.edu/u/v/vss/workspace/research/superpixels/fastSCSP/python/'
    
    with open('./data/coco_images_1000.txt') as f:
        coco_images = f.read().splitlines()
        
    with open('./data/images_random_1000.txt') as f:
        coco_rnd = f.read().splitlines()        
    
    with open('./data/images_uniform_1000.txt') as f:
        coco_uni = f.read().splitlines()           
    
    frame_cnt = 0
    num_sp_iter = 5
    #output_prob_orig = np.zeros((np.size(img_sp_name),np.size(labels)))
    #output_kl = np.zeros(np.size(img_sp_name))
    output_kl = []
    output_dist = []
                                
    for im_idx, im_name in enumerate(coco_images):
    
        #classify the original image:
        image = caffe.io.load_image(im_name)
        net.blobs['data'].data[...] = transformer.preprocess('data', image)

        #plt.figure()
        #plt.imshow(image)
        #plt.show()    
            
        #classify
        net.forward()
    
        # obtain the output probabilities
        output_prob_orig = net.blobs['prob'].data[0].ravel()        
                
        print 'predicted class is:', output_prob_orig.argmax()
        print 'output label:', labels[output_prob_orig.argmax()]

        # sort top five predictions from softmax output
        top_idx = output_prob_orig.argsort()[::-1][:1000]            
                            
        print "top 5 (img idx: %d):" %im_idx
        for idx in top_idx[:5]:
            print str(output_prob_orig[idx]) + ", " + str(labels[idx])
            
        #classify the super-pixel image
        for sp_idx in range(num_sp_iter):
            #uniform segmentation
            im_name_uni = coco_uni[frame_cnt]            
            print frame_cnt
            print im_name_uni
            image_uni = caffe.io.load_image(IMG_PATH + im_name_uni)
            net.blobs['data'].data[...] = transformer.preprocess('data', image_uni)
            net.forward() #classify
            output_prob_uni = 0
            output_prob_uni = net.blobs['prob'].data[0].ravel()        
            print 'uni predicted class is:', output_prob_uni.argmax()
            print 'uni output label:', labels[output_prob_uni.argmax()]            
            top_idx_uni = output_prob_uni.argsort()[::-1][:1000]       
            print "top 5 uni (img idx: %d):" %im_idx
            for idx in top_idx_uni[:5]:
                print str(output_prob_uni[idx]) + ", " + str(labels[idx])
            output_kl.append(compute_kl(output_prob_orig[top_idx[0:100]],output_prob_uni[top_idx_uni[0:100]]))
            output_qq.append(1)
            
            #sampled segmentation            
            im_name_rnd = coco_rnd[frame_cnt]
            image_rnd = caffe.io.load_image(IMG_PATH + im_name_rnd)
            net.blobs['data'].data[...] = transformer.preprocess('data', image_rnd)
            net.forward() #classify
            output_prob_rnd = 0
            output_prob_rnd = net.blobs['prob'].data[0].ravel()        
            print 'rnd predicted class is:', output_prob_rnd.argmax()
            print 'rnd output label:', labels[output_prob_rnd.argmax()]
            top_idx_rnd = output_prob_rnd.argsort()[::-1][:1000]            
            print "top 5 rnd (img idx: %d):" %im_idx
            for idx in top_idx_rnd[:5]:
                print str(output_prob_rnd[idx]) + ", " + str(labels[idx])            
            output_kl.append(compute_kl(output_prob_orig[top_idx[0:100]],output_prob_rnd[top_idx_rnd[0:100]]))
            output_qq.append(1)                        
            
            frame_cnt += 1
        #end
        
    #end 
    
    print "KL: ", output_kl
    
    output_kl_uni = np.array(output_kl[::2]).reshape(len(coco_images), num_sp_iter)
    output_kl_rnd = np.array(output_kl[1::2]).reshape(len(coco_images), num_sp_iter)
   
    """     
    plt.figure()
    plt.plot(np.log(output_prob[0,:]))
    plt.xlabel('ImageNet Categories')
    plt.ylabel('Log probability')
    plt.title('AlexNet Classifier (pre-trained on ImageNet)')
    plt.rcParams.update({'font.size':20})
    plt.grid(True)
    plt.savefig('./figures/output_prob.png')    
    """
    
    #K_real = [12, 42, 143, 550, 2150]
    K_real = range(output_kl_uni.shape[1])

    plt.figure()
    plt.errorbar(K_real,np.mean(output_kl_uni, axis=0), xerr=0, yerr=np.std(output_kl_uni, axis=0), marker = 'o', fmt = '--o', color = 'b', linewidth = 3.0, label = 'uniform seg')
    plt.errorbar(K_real,np.mean(output_kl_rnd, axis=0), xerr=0, yerr=np.std(output_kl_rnd, axis=0), marker = 'o', fmt = '--o', color = 'r', linewidth = 3.0, label = 'sampled seg')
    plt.title('KL vs num SP (AlexNet classifier)')
    plt.xlabel('num superpixels')
    plt.ylabel('KL(p,q)')
    plt.grid(True)
    plt.legend()
    plt.rcParams.update({'font.size':20})
    plt.xticks([])        
    plt.savefig('./figures/output_kl_coco.png')
    
    pdb.set_trace()

if __name__ == "__main__":
    
    main()









