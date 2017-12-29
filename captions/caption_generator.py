import numpy as np
import pandas as pd
import cPickle as pickle

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense
from keras.layers import RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import image, sequence


EMBEDDING_DIM = 128
DATA_PATH = "/data/vision/fisher/data1/Flickr8k/"

class CaptionGenerator():

    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        self.encoded_images = pickle.load(open(DATA_PATH + "encoded_images.dat", "rb"))
        self.variable_initializer()

    def variable_initializer(self):
        df_train = pd.read_csv(DATA_PATH + 'Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        df_val = pd.read_csv(DATA_PATH + 'Flickr8k_text/flickr_8k_val_dataset.txt', delimiter='\t')
        df = pd.concat([df_train, df_val], axis=0)
        df['cap_len'] = df['captions'].apply(lambda words: len(words.split()))

        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        for i in range(nb_samples):
            x = iter.next()
            caps.append(x[1][1])

        self.total_samples=0
        for text in caps:
            self.total_samples+=len(text.split())-1
        print "Total samples : "+str(self.total_samples)
        
        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)
        self.word_index = {}
        self.index_word = {}
        for i, word in enumerate(unique):
            self.word_index[word]=i
            self.index_word[i]=word

        self.max_cap_len = np.int(df['cap_len'].mean() + 2*df['cap_len'].std()) 
        print "Vocabulary size: "+str(self.vocab_size)
        print "Maximum caption length: "+str(self.max_cap_len)
        print "Variables initialization done!"


    def data_generator_train(self, batch_size = 64):
        partial_caps = []
        next_words = []
        images = []
        #print "generating training data..."
        gen_count = 0
        df = pd.read_csv(DATA_PATH + 'Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = iter.next()
            caps.append(x[1][1])
            imgs.append(x[1][0])

        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter+=1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split())-1):
                    total_count+=1
                    partial = [self.word_index[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    next = np.zeros(self.vocab_size)
                    next[self.word_index[text.split()[i+1]]] = 1
                    next_words.append(next)
                    images.append(current_image)
                    
                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        #print "yielding count: "+str(gen_count)
                        #images: CNN encodings 
                        #partial_caps: embedding indices
                        #next_words: one-hot encodings
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                    #end if
                #end for
            #end for
        #end while



    def data_generator_val(self, batch_size = 64):
        partial_caps = []
        next_words = []
        images = []
        #print "generating validation data..."
        gen_count = 0
        df = pd.read_csv(DATA_PATH + 'Flickr8k_text/flickr_8k_val_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = iter.next()
            caps.append(x[1][1])
            imgs.append(x[1][0])

        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter+=1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split())-1):
                    total_count+=1
                    partial = [self.word_index[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    next = np.zeros(self.vocab_size)
                    next[self.word_index[text.split()[i+1]]] = 1
                    next_words.append(next)
                    images.append(current_image)
                    
                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        #print "yielding count: "+str(gen_count)
                        #images: CNN encodings 
                        #partial_caps: embedding indices
                        #next_words: one-hot encodings
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                    #end if
                #end for
            #end for
        #end while
        

    def load_image(self, path):
        img = image.load_img(path, target_size=(224,224))
        x = image.img_to_array(img)
        return np.asarray(x)


    def create_model(self, ret_model = False):

        #image branch
        image_model = Sequential()
        image_model.add(Dense(EMBEDDING_DIM, input_dim = 2048, activation='relu'))
        image_model.add(RepeatVector(self.max_cap_len))

        #text branch
        lang_model = Sequential()
        lang_model.add(Embedding(self.vocab_size, 256, input_length=self.max_cap_len))
        lang_model.add(LSTM(256,return_sequences=True))
        lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

        #concatenated
        model = Sequential()
        model.add(Merge([image_model, lang_model], mode='concat'))
        model.add(LSTM(1024, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))

        print "Model created!"

        if(ret_model==True):
            return model

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model

    def get_word(self,index):
        return self.index_word[index]


