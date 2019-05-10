import data_preprocess, data_preprocess_img
import model
import numpy as np
import os
import config
from evalute_tests import Evaluator
from keras.models import load_model

def show_model_summary():
    video_box_gen = model.VideoBoxClassifier()
    video_box_gen.model.summary()


def videobox_few_training():
    root = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data'
    training_root_folder = os.path.join(root, 'few_training')
    validation_root_folder = os.path.join(root, 'validate')

    video_box_gen = model.VideoBoxClassifier()
    video_box_gen.model.summary()
    dataset = data_preprocess.VideoDataset(training_root_folder, all_in_memory=True,
                                           cache_dir=os.path.join(root, r'cache\training'))

    validation_data = data_preprocess.VideoDataset(validation_root_folder, all_in_memory=True,
                                                   cache_dir=os.path.join(root, r'cache\validate'))
    # video_box_gen.model.fit_generator(dataset, epochs=1000)
    history = video_box_gen.train_all(dataset.x, dataset.y, (validation_data.x, validation_data.y))

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
    root = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data'
    training_root_folder = os.path.join(root, 'train')
    validation_root_folder = os.path.join(root, 'validate')
    video_box_gen = model.VideoBoxClassifier()
    video_box_gen.model.summary()
    dataset = data_preprocess.VideoDataset(training_root_folder, all_in_memory=True,
                                           cache_dir=os.path.join(root, r'cache\training'))

    validation_data = data_preprocess.VideoDataset(validation_root_folder, all_in_memory=True,
                                                   cache_dir=os.path.join(root, r'cache\validate'))

    history = video_box_gen.train_all(dataset.x, dataset.y, (validation_data.x, validation_data.y))


def image_evaluation():
    root = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data'
    test_root_folder = os.path.join(root, 'test')
    validation_root_folder = os.path.join(root, 'validate')
    model_path =  os.path.join(root, r'weights\fewimage_train_weights.20190504235339.h5')
    evaluator = Evaluator(test_root_folder)
    model = load_model(model_path)

    evaluator.make_prediction(model, 'TestResults_11.xlsx')
    dataset = data_preprocess_img.ImageDataset(validation_root_folder,
                                               cache_dir=os.path.join(root, r'cache\validate\not_sorted'), shuffle=False)
    evaluator.measure_model(model, dataset)

def image_few_training():
    root = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data'
    training_root_folder = os.path.join(root, 'train')
    validation_root_folder = os.path.join(root, 'validate')
    img_class_trainer = model.ImageClassifier()
    img_class_trainer.model.summary()

    dataset = data_preprocess_img.ImageDataset(training_root_folder,
                                               cache_dir=os.path.join(root, r'cache\training'))

    validation_data = data_preprocess_img.ImageDataset(validation_root_folder,
                                                       cache_dir=os.path.join(root, r'cache\validate'))
    # video_box_gen.model.fit_generator(dataset, epochs=1000)
    history = img_class_trainer.train_all(dataset.x, dataset.y, (validation_data.x, validation_data.y))
