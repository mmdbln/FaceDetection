import pandas as pd
import shutil
import os
import cv2
from deepface import DeepFace

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]
metrics = ["cosine", "euclidean", "euclidean_l2"]

# read csv as pandas lib
image_dataset = pd.read_csv('/home/ubuntu/Downloads/Telegram Desktop/FR/RF_db.csv')
# directory of our database
database_path = "/home/ubuntu/Downloads/Telegram Desktop/FR/database/"
# directory of the input image
temporary_path = "/home/ubuntu/Downloads/Telegram Desktop/FR/"
# temporary_image = Path(temporary_path).glob('*.jpg')
true_result_found = False
for tem_img in os.listdir(temporary_path):
    tem_img_ = cv2.imread(os.path.join(temporary_path, tem_img)) 

    for image in os.listdir(database_path):
        if not true_result_found:
            db_img = cv2.imread(os.path.join(database_path, image))  

            verifying = DeepFace.verify(img1_path = tem_img_, img2_path = db_img, detector_backend = backends[3], distance_metric=metrics[0])
            result = verifying['verified']

            if result:
                true_result_found = True  
                image_dataset.loc[image_dataset["image name"] == image.split('.')[0], 'crossing times'] += 1

    if not true_result_found:
        image_dataset.loc[len(image_dataset.index)] = [tem_img.split('.')[0], 1]
        shutil.move(os.path.join(temporary_path, tem_img), database_path)

image_dataset.to_csv('/home/ubuntu/Downloads/Telegram Desktop/FR/RF_db.csv')