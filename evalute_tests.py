import xlsxwriter
import numpy as np
import keras
import config
import os
import skvideo
import skimage.transform as trf
from data_preprocess import VideoDataset, Emotions


class Evaluator:

    def __init__(self, test_files_root, dim=(config.sampling_number, config.image_size, config.image_size),
                 n_channels=3, n_classes=6):
        self.root_folder = test_files_root
        self.dim = dim
        self.img_paths = self._get_test_files()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self._prepare_all_test_file()

    def _get_test_files(self):
        img_paths = []
        for subdir, dirs, files in os.walk(self.root_folder):
            for file in files:
                file_absolute_path = os.path.join(subdir, file)
                img_paths.append(file_absolute_path)
        return img_paths

    def _prepare_all_test_file(self):
        X = np.empty((len(self.img_paths), *self.dim, self.n_channels), dtype=np.uint8)

        for i, img_path in enumerate(self.img_paths):
            # Load video
            video_array = VideoDataset.video2memory(img_path)
            # Sample video
            sampled_video = VideoDataset.sample_video(video_array)
            # Store sample
            X[i, ] = sampled_video

        self.x = X

        return self.x

    def make_prediction(self, model, excel_file_name='TestResults.xlsx'):
        workbook = xlsxwriter.Workbook(excel_file_name)
        worksheet = workbook.add_worksheet()
        row = 0
        col = 0
        worksheet.write(row, col, 'File')
        worksheet.write(row, col + 1, 'Label')
        row+=1
        for video_index in range(self.x.shape[0]):
            predicted_emotion_onehot = model.predict(np.asarray([self.x[video_index]]))
            _, file = os.path.split(self.img_paths[video_index])
            file = file.rsplit('.', -1)[0]
            worksheet.write(row, col, file)
            emotion = str(Emotions(np.argmax(predicted_emotion_onehot[0])).name)
            worksheet.write(row, col + 1, emotion)
            row += 1
        workbook.close()


