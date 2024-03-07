import os
import time
import csv
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger(module="commons.realtime")

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# pylint: disable=too-many-nested-blocks


def analysis(
    db_path,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    source=0,
    time_threshold=5,
    frame_threshold=5,
):
    # global variables
    text_color = (255, 255, 255)
    pivot_img_size = 112  # face recognition result image

    enable_emotion = True
    enable_age_gender = True
    # ------------------------
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    model: FacialRecognition = DeepFace.build_model(model_name=model_name)

    # find custom values for this input set
    target_size = model.input_shape

    logger.info(f"facial recognition model {model_name} is just built")
    # -----------------------
    # call a dummy find function for db_path once to create embeddings in the initialization
    DeepFace.find(
        img_path=np.zeros([224, 224, 3]),
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=False,
    )
    # -----------------------
    # visualization
    freeze = False
    face_detected = False
    face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
    freezed_frame = 0
    tic = time.time()
    capture_time = time.time() + 3


    cap = cv2.VideoCapture(source)  # webcam
    while True:
        has_frame, img = cap.read()
        img = cv2.flip(img, 1)
        if not has_frame:
            break

        # cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        raw_img = img.copy()
        resolution_x = img.shape[1]
        resolution_y = img.shape[0]

        if freeze == False:
            try:
                # just extract the regions to highlight in webcam
                face_objs = DeepFace.extract_faces(
                    img_path=img,
                    target_size=target_size,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                )
                faces = []
                for face_obj in face_objs:
                    facial_area = face_obj["facial_area"]
                    #For Test ????????????????????????????????????????
                    if facial_area["w"] <= 130:  # discard small detected faces
                        continue
                    faces.append(
                        (
                            facial_area["x"],
                            facial_area["y"],
                            facial_area["w"],
                            facial_area["h"],
                        )
                    )
            except:  # to avoid exception if no face detected
                faces = []

            if len(faces) == 0:
                face_included_frames = 0
        else:
            faces = []

        detected_faces = []
        face_index = 0
        for x, y, w, h in faces:
            face_detected = True
            if face_index == 0:
                face_included_frames += 1  # increase frame for a single face

            cv2.rectangle(
                img, (x, y), (x + w, y + h), (67, 67, 67), 1
            )  # draw rectangle to main image

            detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]  # crop detected face

            # -------------------------------------

            detected_faces.append((x, y, w, h))
            face_index = face_index + 1

            # -------------------------------------

        if face_detected == True and face_included_frames == frame_threshold and freeze == False:
            # freeze = True
            # base_img = img.copy()
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()
        else:
            cv2.imshow("img", img)

                
        if time.time() >= capture_time:
            cv2.imwrite((f"/home/mohe/Desktop/test/{time.time()}.jpg"),  detected_face)
            detected_face_df = pd.DataFrame(detected_face)
            detected_face_df.to_csv('/home/mohe/Desktop/test/detectedFace.csv', index=False)
            capture_time += 10

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()