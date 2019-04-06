import proceedings
from evalute_tests import Evaluator

# proceedings.videobox_few_training()
proceedings.videobox_all_training()


#
# from keras.models import load_model
# from model import VideoBoxClassifier
# vbclass = VideoBoxClassifier()
#
# vbclass.model = load_model(r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\weights\first_good_model.h5')
#
# evaluator = Evaluator(r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\few_test')
# evaluator.make_prediction(vbclass.model)

# vbclass.model.load_weights(r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\weights\fewtrain_weights.20190405151420.hdf5')
# vbclass.model.save(r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\weights\first_good_model.h5')
# del vbclass.model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#proceedings.videobox_segmentation_trial()



