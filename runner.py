import proceedings
from evalute_tests import Evaluator
from keras.models import load_model

#proceedings.show_model_summary()
#proceedings.videobox_few_training()
#proceedings.videobox_all_training()
#proceedings.image_few_training()

#
# from keras.models import load_model
# from model import VideoBoxClassifier
# vbclass = VideoBoxClassifier()
#
proceedings.image_evaluation()
# vbclass.model.load_weights(r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\weights\fewtrain_weights.20190405151420.hdf5')
# vbclass.model.save(r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\weights\first_good_model.h5')
# del vbclass.model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#proceedings.videobox_segmentation_trial()



