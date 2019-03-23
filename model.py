import keras
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Activation, MaxPooling3D, Flatten

import config

class VideoBoxClassifier:

    def __init__(self):

        model = Sequential()
        model.add(Conv3D(16, (3, 3, 3), padding='same', activation=keras.layers.LeakyReLU(alpha=0.3),
                         input_shape=(config.sampling_number, config.image_size, config.image_size, config.channels),
                         data_format='channels_last'))
        model.add(MaxPooling3D((2, 2, 2)))
        model.add(Conv3D(64, (3, 3, 3), padding='same', activation=keras.layers.LeakyReLU(alpha=0.3)))
        model.add(MaxPooling3D((2, 2, 2)))
        model.add(Conv3D(128, (3, 3, 3), padding='same', activation=keras.layers.LeakyReLU(alpha=0.3)))
        model.add(MaxPooling3D((4, 4, 4)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(len(config.label_dict)))
        model.add(Activation('softmax'))

        self.model = model