
import numpy as np
import matplotlib.pyplot as plt

import os
import random

from PIL import Image
import h5py
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.manifold import TSNE

K.set_image_dim_ordering('th')

def get_image(path):

    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img.getdata(), np.uint8)
    img = img.reshape(224, 224, 3).astype(np.float32)
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)
    
    return img

def VGG_16(weights_path):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))

    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)

    print("finished loading VGG-16 weights...")

    return model

def get_concatenated_images(images, indexes, thumb_height):

    thumbs = []
    for idx in indexes:
        img = Image.open(images[idx])
        img = img.resize((img.width * thumb_height / img.height, thumb_height), Image.ANTIALIAS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)

    return concat_image

def get_closest_images(acts, query_image_idx, num_results=5):

    distances = [distance.euclidean(acts[query_image_idx], act) for act in acts]
    idx_closest  = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
  
    return idx_closest


if __name__ == "__main__":

    vgg_path = "./data/vgg16/vgg16_weights.h5"
    images_path = "./data/101_ObjectCategories"

    num_images = 5000

    model = VGG_16(vgg_path)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    images = [os.path.join(dp,f) for dp, dn, filenames in os.walk(images_path) for f in filenames \
              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]

    if num_images < len(images):
        images = [images[i] for i in sorted(random.sample(xrange(len(images)), num_images))]  

    print('reading %d images...' %len(images))
    
    activations = []
    for idx, image_path in enumerate(images):
        if idx % 10 == 0:
            print('getting activations for %d/%d image...' %(idx,len(images)))
        image = get_image(image_path)
        acts = model.predict(image)
        activations.append(acts)

    f = plt.figure()
    plt.plot(np.array(activations[0]))
    f = plt.savefig('./activations.png')

    # reduce activation dimension
    print('computing PCA...')
    acts = np.concatenate(activations, axis=0)
    pca = PCA(n_components=300)
    pca.fit(acts)
    acts = pca.transform(acts)

    # image search 
    print('image search...') 
    query_image_idx = int(num_images*random.random())
    idx_closest = get_closest_images(acts, query_image_idx)
    query_image = get_concatenated_images(images, [query_image_idx], 300)
    results_image = get_concatenated_images(images, idx_closest, 300)

    f = plt.figure()
    plt.imshow(query_image)
    plt.title("query image (%d)" %query_image_idx)
    f.savefig("./query.png")

    f = plt.figure()
    plt.imshow(results_image)
    plt.title("result images")
    f.savefig("./result_images.png") 











