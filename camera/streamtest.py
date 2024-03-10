import os
import time
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger
from dbcsv import save_csv

logger = Logger(module="commons.realtime")

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# pylint: disable=too-many-nested-blocks

image_dataset = pd.DataFrame(columns=['image name','crossing times'])
def analysis(
    db_path,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=False,
    source=0,
    time_threshold=5,
    frame_threshold=5,
):
    shape = 0
    save = False
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

    if enable_face_analysis:
        DeepFace.build_model(model_name="Age")
        logger.info("Age model is just built")
        DeepFace.build_model(model_name="Gender")
        logger.info("Gender model is just built")
        DeepFace.build_model(model_name="Emotion")
        logger.info("Emotion model is just built")
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
            shape = detected_face.shape[0]
            shape2 = detected_face.shape[1] ##########################################
            save_image = detected_face
            

            # -------------------------------------

            detected_faces.append((x, y, w, h))
            face_index = face_index + 1

            # -------------------------------------

        if face_detected == True and face_included_frames == frame_threshold and freeze == False:
            if shape != shape2:
                freeze = True
            else :
                freeze = False
            # base_img = img.copy()
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()

        if freeze == True:

            toc = time.time()
            if (toc - tic) < time_threshold:

                if freezed_frame == 0:
                    freeze_img = base_img.copy()
                    # here, np.uint8 handles showing white area issue
                    # freeze_img = np.zeros(resolution, np.uint8)
                    

                    for detected_face in detected_faces_final:
                        x = detected_face[0]
                        y = detected_face[1]
                        w = detected_face[2]
                        h = detected_face[3]

                        cv2.rectangle(
                            freeze_img, (x, y), (x + w, y + h), (67, 67, 67), 1
                        )  # draw rectangle to main image

                        # -------------------------------
                        # extract detected face
                        custom_face = base_img[y : y + h, x : x + w]
                        # --------------------------------
                        # face recognition
                        # call find function for custom_face
                        dfs = DeepFace.find(
                            img_path=custom_face,
                            db_path=db_path,
                            model_name=model_name,
                            detector_backend="skip",
                            distance_metric=distance_metric,
                            enforce_detection=False,
                            silent=True,
                        )

                        if len(dfs) > 0:
                            # directly access 1st item because custom face is extracted already
                            df = dfs[0]

                            if df.shape[0] > 0:
                                candidate = df.iloc[0]
                                label = candidate["identity"]

                                # to use this source image as is
                                display_img = cv2.imread(label)
                                # to use extracted face
                                source_objs = DeepFace.extract_faces(
                                    img_path=display_img,
                                    target_size=(pivot_img_size, pivot_img_size),
                                    detector_backend=detector_backend,
                                    enforce_detection=False,
                                    align=False,
                                )

                                if len(source_objs) > 0:
                                    # extract 1st item directly
                                    source_obj = source_objs[0]
                                    display_img = source_obj["face"]
                                    display_img *= 255
                                    display_img = display_img[:, :, ::-1]
                                # --------------------
                                label = label.split("/")[-1]

                                # customize source code:
                                ####################
                                
                                ####################
                                try:
                                    if (
                                        y - pivot_img_size > 0
                                        and x + w + pivot_img_size < resolution_x
                                    ):
                                        # top right
                                        freeze_img[
                                            y - pivot_img_size : y,
                                            x + w : x + w + pivot_img_size,
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x + w, y),
                                            (x + w + pivot_img_size, y + 20),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x + w, y + 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y),
                                            (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                                            (x + w, y - int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )

                                    elif (
                                        y + h + pivot_img_size < resolution_y
                                        and x - pivot_img_size > 0
                                    ):
                                        # bottom left
                                        freeze_img[
                                            y + h : y + h + pivot_img_size,
                                            x - pivot_img_size : x,
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x - pivot_img_size, y + h - 20),
                                            (x, y + h),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x - pivot_img_size, y + h - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y + h),
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (x, y + h + int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )

                                    elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                                        # top left
                                        freeze_img[
                                            y - pivot_img_size : y, x - pivot_img_size : x
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x - pivot_img_size, y),
                                            (x, y + 20),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x - pivot_img_size, y + 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y),
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y - int(pivot_img_size / 2),
                                            ),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (
                                                x + int(w / 2) - int(w / 4),
                                                y - int(pivot_img_size / 2),
                                            ),
                                            (x, y - int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )

                                    elif (
                                        x + w + pivot_img_size < resolution_x
                                        and y + h + pivot_img_size < resolution_y
                                    ):
                                        # bottom righ
                                        freeze_img[
                                            y + h : y + h + pivot_img_size,
                                            x + w : x + w + pivot_img_size,
                                        ] = display_img

                                        overlay = freeze_img.copy()
                                        opacity = 0.4
                                        cv2.rectangle(
                                            freeze_img,
                                            (x + w, y + h - 20),
                                            (x + w + pivot_img_size, y + h),
                                            (46, 200, 255),
                                            cv2.FILLED,
                                        )
                                        cv2.addWeighted(
                                            overlay,
                                            opacity,
                                            freeze_img,
                                            1 - opacity,
                                            0,
                                            freeze_img,
                                        )

                                        cv2.putText(
                                            freeze_img,
                                            label,
                                            (x + w, y + h - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            text_color,
                                            1,
                                        )

                                        # connect face and text
                                        cv2.line(
                                            freeze_img,
                                            (x + int(w / 2), y + h),
                                            (
                                                x + int(w / 2) + int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (67, 67, 67),
                                            1,
                                        )
                                        cv2.line(
                                            freeze_img,
                                            (
                                                x + int(w / 2) + int(w / 4),
                                                y + h + int(pivot_img_size / 2),
                                            ),
                                            (x + w, y + h + int(pivot_img_size / 2)),
                                            (67, 67, 67),
                                            1,
                                        )
                                except Exception as err:  # pylint: disable=broad-except
                                    logger.error(str(err))

                        tic = time.time()  # in this way, freezed image can show 5 seconds

                        # -------------------------------

                time_left = int(time_threshold - (toc - tic) + 1)

                cv2.rectangle(freeze_img, (10, 10), (90, 50), (67, 67, 67), -10)
                cv2.putText(
                    freeze_img,
                    str(time_left),
                    (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow("img", freeze_img)

                freezed_frame = freezed_frame + 1
            else:
                face_detected = False
                face_included_frames = 0
                freeze = False
                freezed_frame = 0

        else:
            cv2.imshow("img", img)

        
        if (shape == shape2) and save:
            time_val = time.time()
            cv2.imwrite((f"/home/mohe/Desktop/db/database/{time_val}.jpg"),  save_image)
            print("saved")
            # time.sleep(4)
            save_csv(csv_path="/home/mohe/Desktop/db/RF_db.csv", db_path="/home/mohe/Desktop/db/database/sefic", saving_path=f"/home/mohe/Desktop/db/database")
            save = False

        if (shape != shape2):
            save = True
            
            

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()

