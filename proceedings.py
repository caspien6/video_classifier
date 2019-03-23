import data_preprocess
import model
import numpy as np

import config

def videobox_few_training():
    root_folder = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\few_training'
    dataset = data_preprocess.VideoDataset(root_folder, all_in_memory=True)

    video_box_gen = model.VideoBoxClassifier()
    history = video_box_gen.train_all(dataset.x, dataset.y)

    # Validation section
    for i in range(dataset.x.shape[0]):
        print(video_box_gen.model.predict(dataset.x[i, np.newaxis, :, :, :, :], batch_size=1))
        print(dataset.y[i])
        print()


def videobox_all_training():
    root_folder = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\train'
    dataset = data_preprocess.VideoDataset(root_folder, batch_size=8, all_in_memory=False)

    video_box_gen = model.VideoBoxClassifier()
    my_model = video_box_gen.model
    my_model.compile('adam', 'binary_crossentropy', ['accuracy'])
    my_model.summary()

    my_model.fit_generator(dataset, epochs=20)

