import caption_generator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "/data/vision/fisher/data1/Flickr8k/"

def train_model(weight = None, batch_size=32, epochs = 10):

    cg = caption_generator.CaptionGenerator()
    model = cg.create_model()

    if weight != None:
        model.load_weights(weight)

    counter = 0
    file_name = DATA_PATH + 'weights-checkpoint.hdf5'

    #define callbacks
    checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    tensor_board = TensorBoard(log_dir='./logs', write_graph=False, write_images=False)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=8, verbose=1, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=16, verbose=1)

    callbacks_list = [checkpoint, tensor_board, reduce_lr, early_stopping]
    hist = model.fit_generator(cg.data_generator(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=epochs, verbose=2, callbacks=callbacks_list)

    try:
        model.save(DATA_PATH + 'final_model.h5', overwrite=True)
        model.save_weights(DATA_PATH + 'final_weights.h5',overwrite=True)
    except:
        print "Error in saving model."

    print "Training complete...\n"

    return hist

if __name__ == '__main__':

    hist = train_model(epochs=128)

    plt.figure()
    plt.plot(hist.history['loss'], label='Adam')
    plt.title('Image Caption Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./captions_training_loss.png')

    plt.figure()
    plt.plot(hist.history['acc'], label='Adam')
    plt.title('Image Caption Model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./captions_training_acc.png')


    #latest_weights = DATA_PATH + '/trained_models/weights-improvement.hdf5'
    #train_model(weight = latest_weights, batch_size=32,  epochs=50)
