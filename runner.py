import data_preprocess
import matplotlib.pyplot as plt
sample_video = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\train\Disgust\trainDisgust002.avi'
root_folder = r'O:\ProgrammingSoftwares\python_projects\video_classifier\data\train'

dataset = data_preprocess.VideoDataset(root_folder)
y = dataset[0][1]

print()