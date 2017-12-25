import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras import backend as K
from keras.applications import vgg19
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array

from time import time
from tqdm import tqdm
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b


def preprocess_image(image_path, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2,0,1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination, img_height, img_width):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x, img_height, img_width):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


CONTENT_IMAGE_PATH = './figures/content_image.jpg'
STYLE_IMAGE_PATH = './figures/style_image.jpg'

width, height = load_img(CONTENT_IMAGE_PATH).size
img_height = 400
img_width = np.int(width * img_height / float(height))

content_image = K.constant(preprocess_image(CONTENT_IMAGE_PATH, img_height, img_width))
style_image = K.constant(preprocess_image(STYLE_IMAGE_PATH, img_height, img_width))
combination_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([content_image, style_image, combination_image], axis=0)
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

total_variation_weight = 1e-4
style_weight = 1.0
content_weight = 0.025

loss = K.variable(0.0)
layer_features = outputs_dict[content_layer]
content_image_features = layer_features[0,:,:,:]
combination_features = layer_features[2,:,:,:]
loss += content_weight * content_loss(content_image_features, combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_image_features = layer_features[1,:,:,:]
    combination_features = layer_features[2,:,:,:]
    sl = style_loss(style_image_features, combination_features, img_height, img_width)
    loss += (style_weight / float(len(style_layers))) * sl

loss += total_variation_weight * total_variation_loss(combination_image, img_height, img_width)

grads = K.gradients(loss, combination_image)[0]
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grad_value = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_value = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_value = grad_value
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_value = np.copy(self.grad_value)
        self.loss_value = None
        self.grad_value = None
        return grad_value

evaluator = Evaluator()

num_iter = 20
x = preprocess_image(CONTENT_IMAGE_PATH, img_height, img_width)
x = x.flatten()
for i in tqdm(range(num_iter)):
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, 
                                     x,
                                     fprime = evaluator.grads,
                                     maxfun=20)
    print "current loss value: ", min_val
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = './figures/style_transfer_iter_%d.png' %i
    imsave(fname, img)
#end for


















