import data_preprocess
import model
import numpy as np
import keras
import os.path as paths
sample_video = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\train\Disgust\trainDisgust002.avi'
weights_folder = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\weights'

def videobox_few_training():
    root_folder = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\few_training'
    dataset = data_preprocess.VideoDataset(root_folder, all_in_memory=True)

    file_path = paths.join(weights_folder, 'weights.{epoch:02d}-{acc:.2f}.hdf5')
    print(file_path)
    checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True,
                                    save_weights_only=True, mode='auto', period=5)

    video_box_gen = model.VideoBoxClassifier()
    my_model = video_box_gen.model
    my_model.compile('adam', 'binary_crossentropy', ['accuracy'])
    my_model.summary()

    history = my_model.fit(dataset.x, dataset.y, batch_size=8, epochs=100, shuffle=False, callbacks=[checkpoint])


    # Validation section
    for i in range(dataset.x.shape[0]):
        print(my_model.predict(dataset.x[i, np.newaxis, :, :, :, :], batch_size=1))
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

