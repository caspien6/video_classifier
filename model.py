from keras.models import Sequential
from keras.layers import Dense, Conv3D, Activation, MaxPooling3D, Flatten, Dropout, BatchNormalization
import keras
import os.path as paths
import datetime
import config
import numpy as np

class VideoBoxClassifier:

    def __init__(self):
        self._create_model()
        self.start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    def _create_model(self):
        model = Sequential()
        model.add(Conv3D(32, (3, 3, 3), strides=(1,1,1),padding='same', activation='relu', kernel_initializer=keras.initializers.lecun_normal(seed=12),
                         input_shape=(config.sampling_number, config.image_size, config.image_size, config.channels),
                         data_format='channels_last'))
        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2),padding='same',  activation='relu',
                         kernel_initializer=keras.initializers.lecun_normal(seed=12)))
        model.add(BatchNormalization())
        model.add(MaxPooling3D((1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1),padding='same',  activation='relu',
                         kernel_initializer=keras.initializers.lecun_normal(seed=12)))
        model.add(Conv3D(64, (3, 3, 3), strides=(1, 2, 2),padding='same',  activation='relu',
                         kernel_initializer=keras.initializers.lecun_normal(seed=12)))
        model.add(BatchNormalization())
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
        model.add(Flatten())
        model.add(Dense(512,activation='relu', kernel_initializer=keras.initializers.lecun_normal(seed=12)))
        #model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu', kernel_initializer=keras.initializers.lecun_normal(seed=12)))
        #model.add(Dropout(0.25))
        model.add(Dense(len(config.label_dict), activation='softmax', kernel_initializer=keras.initializers.lecun_normal(seed=12)))

        self.model = model
        optimizer = keras.optimizers.Adam(lr=1e-6)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def _get_callbacks(self):
        file_path = paths.join(config.weights_folder, 'fewtrain_weights.' + self.start_time + '.h5')
        print(file_path)
        checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True,
                                                     save_weights_only=False, mode='auto', period=1)
        tfboard = keras.callbacks.TensorBoard(log_dir='./logs/train_' + self.start_time, histogram_freq=1, batch_size=2, write_graph=True)
        return [checkpoint, tfboard]

    def train_all(self, x, y):
        callbacks = self._get_callbacks()
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        x = x[indices]
        y = y[indices]
        # Returns history object
        return self.model.fit(x, y, batch_size=1, epochs=1000, shuffle=True, callbacks=callbacks, validation_split=0.1)


    # first good model 0.55 accuracy...
    # def _create_model(self):
    #     model = Sequential()
    #     model.add(Conv3D(32, (3, 3, 3), strides=(2,2,2),padding='same', activation='relu', kernel_initializer=keras.initializers.lecun_normal(seed=12),
    #                      input_shape=(config.sampling_number, config.image_size, config.image_size, config.channels),
    #                      data_format='channels_last'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling3D((1, 2, 2), strides=(1, 2, 2)))
    #     model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1),padding='same',  activation='relu',
    #                      kernel_initializer=keras.initializers.lecun_normal(seed=12)))
    #     model.add(Conv3D(64, (3, 3, 3), strides=(1, 2, 2),padding='same',  activation='relu',
    #                      kernel_initializer=keras.initializers.lecun_normal(seed=12)))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    #     model.add(BatchNormalization())
    #     model.add(Flatten())
    #     model.add(Dense(256,activation='relu', kernel_initializer=keras.initializers.lecun_normal(seed=12)))
    #     #model.add(Dropout(0.25))
    #     model.add(Dense(128, activation='relu', kernel_initializer=keras.initializers.lecun_normal(seed=12)))
    #     #model.add(Dropout(0.25))
    #     model.add(Dense(len(config.label_dict), activation='softmax', kernel_initializer=keras.initializers.lecun_normal(seed=12)))
    #
    #     self.model = model
    #     optimizer = keras.optimizers.Adam(lr=1e-6)
    #     self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])