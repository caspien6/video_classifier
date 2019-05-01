import proceedings
from evalute_tests import Evaluator

#proceedings.show_model_summary()
#proceedings.videobox_few_training()
#proceedings.videobox_all_training()
proceedings.image_few_training()

#
# from keras.models import load_model
# from model import VideoBoxClassifier
# vbclass = VideoBoxClassifier()
#
#
# evaluator = Evaluator(r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\test')
# vbclass.model = load_model(r'O:\ProgrammingSoftwares\python_projects\video_classifier\results\nyolcadik_futtats\eight_training.h5')
# evaluator.make_prediction(vbclass.model, 'TestResults_8.xlsx')

# vbclass.model.load_weights(r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\weights\fewtrain_weights.20190405151420.hdf5')
# vbclass.model.save(r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\weights\first_good_model.h5')
# del vbclass.model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#proceedings.videobox_segmentation_trial()



