import data_preprocess
import model
import numpy as np

import config

def videobox_few_training():
    root_folder = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\few_training'
    video_box_gen = model.VideoBoxClassifier()
    video_box_gen.model.summary()
    dataset = data_preprocess.VideoDataset(root_folder, all_in_memory=True)


    #video_box_gen.model.fit_generator(dataset, epochs=1000)
    history = video_box_gen.train_all(dataset.x, dataset.y)

    # Validation section
    for i in range(dataset.x.shape[0]):
        output = video_box_gen.model.predict(dataset.x[i, np.newaxis, :, :, :, :], batch_size=1)
        print(output)
        print(dataset.y[i])
        print()

def videobox_segmentation_trial():
    root_folder = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\few_training'
    dataset = data_preprocess.VideoDataset(root_folder, all_in_memory=False)
    dataset.try_segmentation()


def videobox_all_training():
    root_folder = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\train'
    video_box_gen = model.VideoBoxClassifier()
    video_box_gen.model.summary()
    dataset = data_preprocess.VideoDataset(root_folder, all_in_memory=True)

    history = video_box_gen.train_all(dataset.x, dataset.y)

    # Validation section
    for i in range(dataset.x.shape[0]):
        output = video_box_gen.model.predict(dataset.x[i, np.newaxis, :, :, :, :], batch_size=1)
        print(output)
        print(dataset.y[i])
        print()

