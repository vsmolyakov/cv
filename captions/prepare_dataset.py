import numpy as np 
import cPickle as pickle
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications import imagenet_utils

from tqdm import tqdm
from time import time

counter = 0
DATA_PATH = "/data/vision/fisher/data1/Flickr8k/"

def load_image(path):
    img = image.load_img(path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)
    return np.asarray(x)

def load_encoding_model():
	model = ResNet50(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
	return model

def get_encoding(model, img):
	global counter
	counter += 1
	image = load_image(DATA_PATH + 'Flicker8k_Dataset/'+str(img))
	pred = model.predict(image)
	pred = np.reshape(pred, pred.shape[-1])
	return pred

def prepare_dataset(no_imgs = -1, num_val=500):
	f_train_images = open(DATA_PATH + 'Flickr8k_text/Flickr_8k.trainImages.txt','rb')
	train_imgs = f_train_images.read().strip().split('\n') if no_imgs == -1 else f_train_images.read().strip().split('\n')[:no_imgs]
	f_train_images.close()

	f_test_images = open(DATA_PATH + 'Flickr8k_text/Flickr_8k.testImages.txt','rb')
	test_imgs = f_test_images.read().strip().split('\n') if no_imgs == -1 else f_test_images.read().strip().split('\n')[:no_imgs]
	f_test_images.close()

	f_train_dataset = open(DATA_PATH + 'Flickr8k_text/flickr_8k_train_dataset.txt','wb')
	f_train_dataset.write("image_id\tcaptions\n")

	f_val_dataset = open(DATA_PATH + 'Flickr8k_text/flickr_8k_val_dataset.txt','wb')
	f_val_dataset.write("image_id\tcaptions\n")

	f_test_dataset = open(DATA_PATH + 'Flickr8k_text/flickr_8k_test_dataset.txt','wb')
	f_test_dataset.write("image_id\tcaptions\n")

	f_captions = open(DATA_PATH + 'Flickr8k_text/Flickr8k.token.txt', 'rb')
	captions = f_captions.read().strip().split('\n')
	data = {}
	print "processing captions..."
	for row in captions:
		row = row.split("\t")
		row[0] = row[0][:len(row[0])-2]
		try:
			data[row[0]].append(row[1])
		except:
			data[row[0]] = [row[1]]
	f_captions.close()

	encoded_images = {}
	encoding_model = load_encoding_model()

	c_train, c_val = 0, 0
	print "processing training and validation images..."
	for idx, img in tqdm(enumerate(train_imgs)):
		encoded_images[img] = get_encoding(encoding_model, img)
		if (idx < len(train_imgs) - num_val):  #training
  		    for capt in data[img]:
			    caption = "<start> "+capt+" <end>"
			    f_train_dataset.write(img+"\t"+caption+"\n")
			    f_train_dataset.flush()
			    c_train += 1
			#end for
		else:  #validation
  		    for capt in data[img]:
			    caption = "<start> "+capt+" <end>"
			    f_val_dataset.write(img+"\t"+caption+"\n")
			    f_val_dataset.flush()
			    c_val += 1
			#end for
	    #end if
    #end for

	f_train_dataset.close()
	f_val_dataset.close()

	c_test = 0
	print "processing test images..."
	for img in tqdm(test_imgs):
		encoded_images[img] = get_encoding(encoding_model, img)
		for capt in data[img]:
			caption = "<start> "+capt+" <end>"
			f_test_dataset.write(img+"\t"+caption+"\n")
			f_test_dataset.flush()
			c_test += 1
	f_test_dataset.close()
	with open(DATA_PATH + "encoded_images.dat", "wb" ) as pickle_f:
		pickle.dump( encoded_images, pickle_f )  
	return [c_train, c_val, c_test]

if __name__ == '__main__':
	c_train, c_val, c_test = prepare_dataset()
	print "num training captions: ", c_train
	print "num validation captions: ", c_val
	print "num test captions:  ", c_test


