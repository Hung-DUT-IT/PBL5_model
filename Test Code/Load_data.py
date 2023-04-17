import cv2 as cv
import numpy as np
import os
from Preprocessing.Detect_face import DETECTION
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class LOAD_DATA:
    def __init__(self):
        self.detector = DETECTION()
        self.X_train = []
        self.Y_label = []    

    def load_faces(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DataSet_dir = BASE_DIR.replace("Preprocessing" , "DataSet")
        Raw_img_dir = os.path.join(DataSet_dir, "raw")

        for root, dirs, files in os.walk(Raw_img_dir):
            for file in files:
                if file.endswith("mp4"):
                    path = os.path.join(root, file)
                    label = os.path.basename(root)
                  
                    single_faces = self.detector.extract_face(path)

                    labels = [label for _ in range(len(single_faces))]
                    
                    self.X_train.extend(single_faces)
                    self.Y_label.extend(labels)

                    directory_path = os.path.join( Raw_img_dir.replace("raw", "processed"), label)
                    os.makedirs(directory_path, exist_ok=True)

                    for i, face in enumerate(single_faces):
                        new_filename = os.path.join( directory_path, os.path.splitext(file)[0]) + str(i) + ".jpg"

                        cv.imwrite(new_filename, face)

        return np.asarray(self.X_train), np.asarray(self.Y_label)
