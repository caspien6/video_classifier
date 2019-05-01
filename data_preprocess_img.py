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
from img_diffuser import anisodiff
from moviepy.editor import *
from data_preprocess import Emotions
import copy


class ImageDataset(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root_folder, batch_size=4, dim=(config.image_size, config.image_size),
                 n_channels=1, n_classes=6, shuffle=True,
                 cache_dir=r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\cache\training'):
        'Initialization'
        self.root_folder = root_folder
        self.dim = dim
        self.batch_size = batch_size
        self.labels = None
        self.cache_dir = cache_dir
        self.img_paths = None
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self._prepare_training_data()
        self.input_filename = 'sampled_images.npy'
        self.output_filename = 'image_labels.npy'

        if not os.path.isfile(os.path.join(cache_dir, self.input_filename)) or \
                not os.path.isfile(os.path.join(cache_dir, self.output_filename)):
            self._prepare_all_data()
        else:
            self._load_from_cache()
        self.on_epoch_end()

    def _load_from_cache(self):
        print('Loading data from cache...')
        self.x = np.load(os.path.join(self.cache_dir, self.input_filename))
        self.y = np.load(os.path.join(self.cache_dir, self.output_filename))

    def _prepare_training_data(self):
        classes = copy.deepcopy(config.label_dict)
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
        #X = np.empty((len(self.img_paths)*config.sampling_number, *self.dim, self.n_channels), dtype=np.uint8)
        #y = np.empty((len(self.img_paths)*config.sampling_number), dtype=np.float)
        x_list = []
        y_list = []
        for i, img_path in enumerate(self.img_paths):

            with VideoFileClip(img_path) as video:
                audio_array = video.audio.to_soundarray().sum(axis=1)/2
            # Load video
            video_array = self.video2memory(img_path)
            # Sample video
            sampled_video = self.sample_video(video_array, audio_array)
            if sampled_video is None: continue
            # Store sample
            x_list.append(sampled_video)
            # Store class
            for j in range(config.sampling_number):
                y_list.append(self.labels[i].value)
        #plt.show()
        self.x = np.asarray(x_list, dtype=np.uint8).reshape((-1,config.image_size,config.image_size,1))
        self.y = keras.utils.to_categorical(np.asarray(y_list, dtype=np.uint8), num_classes=self.n_classes)
        #self.x /= np.max(np.abs(self.x), axis=0)

        np.save(os.path.join(self.cache_dir, self.input_filename), self.x)
        np.save(os.path.join(self.cache_dir, self.output_filename), self.y)
        return self.x, self.y

    @staticmethod
    def sample_video(video_array, audio_array):
        # plt.plot(audio_array)
        # plt.show()
        number_of_frame = video_array.shape[0]
        sorted_indices = np.rint(np.argsort(audio_array) / (len(audio_array) / video_array.shape[0]))
        uniq_indexes = np.unique(sorted_indices, return_index=True)[1]
        sorted_indices = np.asarray([sorted_indices[index] for index in sorted(uniq_indexes)], dtype=np.uint8)
        sorted_indices = sorted_indices[:min(config.sampling_number*2, number_of_frame)]
        indexes = sorted_indices
        # np.random.shuffle(indexes)
        # indexes = np.sort(indexes)
        face_cascade = cv2.CascadeClassifier(
            r'O:\ProgrammingSoftwares\python_projects\video_classifier\video_class_venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

        sampled_video_list = []
        chosen_indices = []
        i = 0
        while len(sampled_video_list) < config.sampling_number:
            try:
                video_frame = trf.resize(video_array[indexes[i]], (config.haar_image_resize, config.haar_image_resize),
                                         preserve_range=True)
                video_frame = np.asarray(video_frame, np.uint8)
                face_candidates = add_haar_face_info(video_frame, face_cascade)
                if len(face_candidates) == 0:
                    i += 1
                    continue
                if face_candidates.any():
                    video_face_box = ImageDataset.crop_face_from_video(video_frame, face_candidates)
                    sampled_video_list.append(video_face_box)
                    chosen_indices.append(indexes[i])
            except IndexError as ie:
                print('Index error!!')
                print(ie)
                return
            except Exception as e:
                i += 1
                print(e)
                continue
            i += 1

        zipped = zip(chosen_indices, sampled_video_list)
        zipped = sorted(zipped, key=lambda l: l[0])
        idxs, sampled_video_list = zip(*zipped)
        random_image_frames = np.asarray(sampled_video_list, dtype=np.uint8)

        return random_image_frames

    @staticmethod
    def crop_face_from_video(video_frame, face_candidates):
        x, y, w, h = face_candidates[0]
        # visualize_haar_box()
        x = max(0, x - config.haar_border_zoom)
        x2 = min(config.haar_image_resize, x + w + config.haar_border_zoom * 2)
        y = max(0, y - config.haar_border_zoom)
        y2 = min(config.haar_image_resize, y + h + config.haar_border_zoom * 2)

        video_face_box = trf.resize(video_frame[y:y2, x:x2], (config.image_size, config.image_size),
                                    preserve_range=True)
        video_face_box = np.asarray(video_face_box, np.uint8)
        return video_face_box

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_paths)*config.sampling_number / self.batch_size))

    def __getitem__(self, index):
        print('Get item nincs implementálva')
        return -1

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths)*config.sampling_number)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @staticmethod
    def video2memory(path_to_file):
        return skvideo.io.vread(path_to_file, as_grey=True)

    def visualize_haar_box(self, img, facebox):
        for (x, y, w, h) in facebox:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def add_haar_face_info(image, face_cascade):
    if image.shape[-1] != 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    face_candidates = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80), maxSize=(140, 140))
    return face_candidates

