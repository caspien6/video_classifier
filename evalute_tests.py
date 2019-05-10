import operator
import xlsxwriter
import numpy as np
import keras
import config
import os
import skvideo
import skimage.transform as trf
from data_preprocess import VideoDataset, Emotions
from data_preprocess_img import VideoPreprocessor

class Evaluator:

    def __init__(self, test_files_root, dim=(config.image_size, config.image_size),
                 n_channels=1, n_classes=6):
        self.root_folder = test_files_root
        self.dim = dim
        self.img_paths = self._get_test_files()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.cache_dir = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\cache\test'
        self.test_filename = 'sampled_test_files.npy'
        self.preprocessor = VideoPreprocessor()

        if not os.path.isfile(os.path.join(self.cache_dir, self.test_filename)):
            self._prepare_all_test_file()
        else:
            self._load_from_cache()

    def _load_from_cache(self):
        print('Loading data from cache...')
        self.x = np.load(os.path.join(self.cache_dir, self.test_filename))

    def _get_test_files(self):
        img_paths = []
        for subdir, dirs, files in os.walk(self.root_folder):
            for file in files:
                file_absolute_path = os.path.join(subdir, file)
                img_paths.append(file_absolute_path)
        return img_paths

    def _prepare_all_test_file(self):
        x_list = []

        for i, img_path in enumerate(self.img_paths):
            sampled_video = self.preprocessor.preprocess_video(img_path)
            if sampled_video is None:
                x_list.append(np.random.rand(config.sampling_number, config.image_size, config.image_size,1))
            # Store sample
            x_list.append(sampled_video)

        self.x = np.asarray(x_list, dtype=np.uint8)
        np.save(os.path.join(self.cache_dir, self.test_filename), self.x)
        return self.x

    def measure_model(self, model, dataset):
        list_of_metrics = model.evaluate(dataset.x, dataset.y)
        print('Metrics: {0} {1}'.format(list_of_metrics, model.metrics_names))

        good_count, bad_count, all_count = (0, 0, 0)
        for video_index in range(0, dataset.x.shape[0], config.sampling_number):
            emotion_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for frame_shift in range(config.sampling_number):
                predicted_emotion_onehot = model.predict(np.asarray([dataset.x[video_index + frame_shift]]))
                emotion_dict[np.argmax(predicted_emotion_onehot[0])] += 1
            emotion_num = self._find_biggest_dict_value(emotion_dict)
            real_emotion_num = np.argmax(dataset.y[video_index])

            if emotion_num == real_emotion_num:
                good_count += 1
            else:
                bad_count += 1

            all_count += 1

        print('Összes videó: {0}, ebből eltalált: {1} darabot, elhibázott: {2} darabot\nAz accuracy: {3}'
              .format(all_count, good_count, bad_count, good_count/all_count))

    def make_prediction(self, model, excel_file_name='TestResults.xlsx'):
        workbook = xlsxwriter.Workbook(excel_file_name)
        worksheet = workbook.add_worksheet()
        row = 0
        col = 0
        worksheet.write(row, col, 'File')
        worksheet.write(row, col + 1, 'Label')
        row += 1
        for video_index in range(self.x.shape[0]):
            emotion_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for frame_index in range(self.x.shape[1]):
                predicted_emotion_onehot = model.predict(np.asarray([self.x[video_index, frame_index]]))
                emotion_dict[np.argmax(predicted_emotion_onehot[0])] += 1

            emotion_num = self._find_biggest_dict_value(emotion_dict)
            _, file = os.path.split(self.img_paths[video_index])
            file = file.rsplit('.', -1)[0]
            worksheet.write(row, col, file)
            emotion = str(Emotions(emotion_num).name)
            worksheet.write(row, col + 1, emotion)
            row += 1
        workbook.close()

    def _find_biggest_dict_value(self, emotion_dict):
        return max(emotion_dict.items(), key=operator.itemgetter(1))[0]

