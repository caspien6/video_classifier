from keras.models import Sequential
from keras.layers import Dense, Conv3D, Activation, MaxPooling3D, Flatten
import keras
import os.path as paths
import datetime
import config

class VideoBoxClassifier:

    def __init__(self):
        self._create_model()
        self.start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    def _create_model(self):
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
        self.model.compile('adam', 'binary_crossentropy', ['accuracy'])

    def _get_callbacks(self):
        file_path = paths.join(config.weights_folder, 'fewtrain_weights.' + self.start_time + '.hdf5')
        print(file_path)
        checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True,
                                                     save_weights_only=True, mode='auto', period=5)
        return [checkpoint]

    def train_all(self, x, y):
        callbacks = self._get_callbacks()

        # Returns history object
        return self.model.fit(x, y, batch_size=8, epochs=100, shuffle=False, callbacks=callbacks)
