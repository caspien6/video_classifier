import skvideo.io
import skvideo.datasets
import os
import config
from enum import Enum
import numpy as np
import keras
import skimage.transform as trf

class Emotions(Enum):
    Anger = 0
    Disgust = 1
    Fear = 2
    Happiness = 3
    Sadness = 4
    Surprise = 5

class VideoDataset(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root_folder, batch_size=4, dim=(config.sampling_number, config.image_size, config.image_size),
                 n_channels=3, n_classes=6, shuffle=True, all_in_memory = False):
        'Initialization'
        self.root_folder = root_folder
        self.dim = dim
        self.batch_size = batch_size
        self.labels = None
        self.img_paths = None
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.all_in_memory = all_in_memory
        self._prepare_training_data()
        self.on_epoch_end()

    def _prepare_training_data(self):
        classes = config.label_dict
        for subdir, dirs, files in os.walk(self.root_folder):
            for file in files:
                file_absolute_path = os.path.join(subdir, file)
                label_name = subdir.rsplit(os.sep, 1)[1]
                classes[label_name].append(file_absolute_path)

        self._prepare_batch_training(classes)

        if self.all_in_memory:
            self._prepare_all_data()

    def _prepare_batch_training(self, classes):
        img_paths = []
        labels = []
        for key, value in classes.items():
            for file_path in value:
                img_paths.append(file_path)
                labels.append(Emotions[key])

        self.img_paths = img_paths
        self.labels = labels

    def _prepare_all_data(self):
        imgs = []
        X = np.empty((len(self.img_paths), *self.dim, self.n_channels))
        y = np.empty((len(self.img_paths)), dtype=int)

        for i, img_path in enumerate(self.img_paths):
            # Load video
            video_array = self._video2memory(img_path)
            # Sample video
            sampled_video = self._sample_video(video_array)

            # Store sample
            X[i, ] = sampled_video

            # Store class
            y[i] = self.labels[i].value

        self.x = X
        self.y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return self.x, self.y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs and labels
        image_paths = [self.img_paths[k] for k in indexes]
        image_labels = [self.labels[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(image_paths, image_labels)

        return x, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_paths, image_labels):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, img_path in enumerate(image_paths):
            # Load video
            video_array = self._video2memory(img_path)
            # Sample video
            sampled_video = self._sample_video(video_array)

            # Store sample
            X[i, ] = sampled_video

            # Store class
            y[i] = image_labels[i].value

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def _video2memory(self, path_to_file):
        return skvideo.io.vread(path_to_file)

    def _sample_video(self, video_array):
        indexes = np.arange(video_array.shape[0])  # 0 is the frame dimension.
        np.random.shuffle(indexes)

        # Find list of frames
        random_image_frames = [trf.resize(video_array[k], (config.image_size, config.image_size))
                               for k in indexes[:config.sampling_number]]
        random_image_frames = np.asarray(random_image_frames)
        return random_image_frames

