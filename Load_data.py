import cv2 as cv
import numpy as np
import os
from Detect_face import DETECTION

class LOAD_DATA:
    def __init__(self):
        self.detector = DETECTION()
        self.X_train = []
        self.Y_label = []

    def load_faces(self):
        label_ids = {}
        current_id = 0 
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))      
        DataSet_dir = os.path.join(BASE_DIR, "DataSet")
        Raw_img_dir = os.path.join(DataSet_dir, "raw")
        
        for root, dirs, files in os.walk(Raw_img_dir):
            for file in files:
                if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png"):
                    path = os.path.join(root, file )                           
                    label = os.path.basename(root)
                    
                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    _id = label_ids[label]
                    single_face = self.detector.extract_face(path)

                    self.X_train.append(single_face)
                    self.Y_label.append(_id)

                    directory_path = os.path.join(Raw_img_dir.replace("raw","processed"), label)
                    os.makedirs(directory_path, exist_ok=True)
                    cv.imwrite(os.path.join(directory_path, os.path.basename(file)), single_face)
                    
        return np.asarray(self.X_train), np.asarray(self.Y_label)
