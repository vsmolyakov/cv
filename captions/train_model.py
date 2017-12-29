import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import caption_generator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler 

import keras
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import load_model

import cPickle as pickle

sns.set_style("whitegrid")

DATA_PATH = "/data/vision/fisher/data1/Flickr8k/"

def step_decay(epoch):
    lr_init = 0.001
    drop = 0.5
    epochs_drop = 4.0
    lr_new = lr_init * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr_new

class LR_hist(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))


def train_model(weight = None, batch_size=256, epochs = 10):

    cg = caption_generator.CaptionGenerator()
    model = cg.create_model()

    if weight != None:
        model.load_weights(weight)

    counter = 0
    file_name = DATA_PATH + 'weights-checkpoint.h5'

    #define callbacks
    checkpoint = ModelCheckpoint(file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensor_board = TensorBoard(log_dir='./logs', write_graph=True)
    hist_lr = LR_hist()
    reduce_lr = LearningRateScheduler(step_decay) 
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=16, verbose=1)
    callbacks_list = [checkpoint, tensor_board, hist_lr, reduce_lr, early_stopping]

    hist = model.fit_generator(cg.data_generator_train(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=epochs, verbose=2, callbacks=callbacks_list, validation_data=cg.data_generator_val(batch_size=batch_size), validation_steps=cg.total_samples/(batch_size*13.0))

    model.save(DATA_PATH + 'final_model.h5', overwrite=True)
    model.save_weights(DATA_PATH + 'final_weights.h5',overwrite=True)

    hist_file = DATA_PATH + '/hist_model.dat'
    with open(hist_file, 'w') as f:
        pickle.dump(hist.history, f)

    print "training complete...\n"

    return model, hist, hist_lr

if __name__ == '__main__':

    model, hist, hist_lr = train_model(epochs=32)

    #generate plots
    plt.figure()
    plt.plot(hist.history['loss'], c='b', lw=2.0, label='train')
    plt.plot(hist.history['val_loss'], c='r', lw=2.0, label='val')
    plt.title('Image Caption Model')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend(loc='upper right')
    plt.savefig('./figures/captions_training_loss.png')

    plt.figure()
    plt.plot(hist.history['acc'], c='b', lw=2.0, label='train')
    plt.plot(hist.history['val_acc'], c='r', lw=2.0, label='val')
    plt.title('Image Caption Model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.savefig('./figures/captions_training_acc.png')

    plt.figure()
    plt.plot(hist_lr.lr, lw=2.0, label='learning rate')
    plt.title('Image Caption Model')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.savefig('./figures/captions_learning_rate.png')

    plot_model(model, show_shapes=True, to_file='./figures/captions_model.png')

