import skvideo.io
import skvideo.datasets
import os
import config
from enum import Enum
import numpy as np
import keras
import skimage.transform as trf
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
                 n_channels=3, n_classes=6, shuffle=True, all_in_memory=False):
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
        if self.all_in_memory:
            self._prepare_all_data()
        self.on_epoch_end()

    def _prepare_training_data(self):
        classes = config.label_dict
        for subdir, dirs, files in os.walk(self.root_folder):
            for file in files:
                file_absolute_path = os.path.join(subdir, file)
                label_name = subdir.rsplit(os.sep, 1)[1]
                classes[label_name].append(file_absolute_path)

        self._prepare_batch_training(classes)

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
        X = np.empty((len(self.img_paths), *self.dim, self.n_channels), dtype=np.uint8)
        y = np.empty((len(self.img_paths)), dtype=np.float)

        for i, img_path in enumerate(self.img_paths):
            # Load video
            video_array = self.video2memory(img_path)
            # Sample video
            sampled_video = self.sample_video(video_array)

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

    def __data_generation(self, video_paths, image_labels):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=np.float)

        # Generate data
        for i, video_path in enumerate(video_paths):
            # Load video
            video_array = self._video2memory(video_path)
            # Sample video
            sampled_video = self.old_sample_video(video_array)

            # Store sample
            X[i, ] = sampled_video

            # Store class
            y[i] = image_labels[i].value

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    @staticmethod
    def video2memory(path_to_file):
        return skvideo.io.vread(path_to_file)

    @staticmethod
    def old_sample_video(video_array):
        indexes = np.arange(video_array.shape[0])  # 0 is the frame dimension.
        np.random.shuffle(indexes)
        indexes = np.sort(indexes[:config.sampling_number])
        # Find list of frames
        random_image_frames = [trf.resize(video_array[k], (config.image_size, config.image_size), preserve_range=True)
                               for k in indexes]
        random_image_frames = np.asarray(random_image_frames, dtype=np.uint8)

        return random_image_frames

    @staticmethod
    def sample_video(video_array):
        indexes = np.arange(video_array.shape[0])  # 0 is the frame dimension.
        np.random.shuffle(indexes)
        #  indexes = np.sort(indexes)
        face_cascade = cv2.CascadeClassifier(r'O:\ProgrammingSoftwares\python_projects\video_classifier\video_class_venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

        sampled_video_list = []
        chosen_indices = []
        i = 0
        while len(sampled_video_list) < config.sampling_number:
            video_frame = trf.resize(video_array[indexes[i]], (config.haar_image_resize, config.haar_image_resize),
                                     preserve_range=True)
            video_frame = np.asarray(video_frame, np.uint8)
            face_candidates = add_haar_face_info(video_frame, face_cascade)
            if len(face_candidates) == 0:
                i += 1
                continue
            if face_candidates.any():
                x, y, w, h = face_candidates[0]
                #visualize_haar_box()
                x = max(0, x-config.haar_border_zoom)
                x2 = min(config.haar_image_resize, x + w + config.haar_border_zoom*2)
                y = max(0, y - config.haar_border_zoom)
                y2 = min(config.haar_image_resize, y + h + config.haar_border_zoom*2)

                video_face_box = trf.resize(video_frame[y:y2, x:x2], (config.image_size, config.image_size),
                           preserve_range=True)
                video_face_box = np.asarray(video_face_box, np.uint8)
                sampled_video_list.append(video_face_box)
                chosen_indices.append(indexes[i])
            i += 1

        zipped = zip(chosen_indices, sampled_video_list)
        zipped = sorted(zipped, key = lambda l: l[0])
        idxs, sampled_video_list = zip(*zipped)
        random_image_frames = np.asarray(sampled_video_list, dtype=np.uint8)

        return random_image_frames

    def visualize_haar_box(self, img, facebox):
        for (x, y, w, h) in facebox:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def try_segmentation(self):

        img_path = self.img_paths[1]
        video_array = self._video2memory(img_path)
        img = np.asarray(video_array[60], dtype=int)
        hist, bin_edges = np.histogram(img, bins=60)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        binary_img = img > 120

        plt.figure(figsize=(11, 4))

        plt.subplot(131)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(132)
        plt.plot(bin_centers, hist, lw=2)
        plt.axvline(120, color='r', ls='--', lw=2)
        plt.text(0.57, 0.8, 'histogram', fontsize=20, transform=plt.gca().transAxes)
        plt.yticks([])
        plt.subplot(133)
        plt.imshow(binary_img[:, :, 0],cmap=plt.cm.gray, interpolation='nearest')
        plt.axis('off')

        plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
        plt.show()


def add_haar_face_info(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_candidates = face_cascade.detectMultiScale(gray, 1.02, 4, minSize=(60, 60), maxSize=(140, 140))
    return face_candidates

