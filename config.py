label_dict = {
            'Anger': [],
            'Disgust': [],
            'Fear': [],
            'Happiness': [],
            'Sadness': [],
            'Surprise': [],

        }

sampling_number = 30
image_size = 128
haar_image_resize = 240
haar_border_zoom = 10 # in pixel
channels = 3

sample_video = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\train\Disgust\trainDisgust002.avi'
weights_folder = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\weights'