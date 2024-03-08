
import cv2
import pandas as pd
from deepface import DeepFace 
import os
import pandas as pd

face_dataset = pd.DataFrame(columns=['file_name','nums'])
# I just added infos of a picture to the dataframe
face_dataset.loc[len(face_dataset.index)] = ["1709877523.7602088.jpg",0]

metrics = ["cosine", "euclidean", "euclidean_l2"]
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

# directory of our database
folder = "/home/ubuntu/Mapsa-ML/ml-advance/project/new_file/"
# directory of the input image
new_img = "16.jpg"
new_im_path = os.path.join("/home/ubuntu/Mapsa-ML/ml-advance/project/" ,new_img)
true_result_found = False

for filename in os.listdir(folder):
    if not true_result_found: 
        input_img = cv2.imread(new_im_path) 
        database_img = cv2.imread(os.path.join(folder,filename))    
        verifying = DeepFace.verify(img1_path = new_im_path, img2_path = database_img,detector_backend = backends[8],distance_metric=metrics[2])
        result = verifying['verified']
        if result:
            true_result_found = True  
            face_dataset.loc[face_dataset["file_name"]==filename,"nums"] += 1

if not true_result_found:
    face_dataset.loc[len(face_dataset.index)] = [new_img,1]
    cv2.imwrite(os.path.join(folder,new_img), input_img)

