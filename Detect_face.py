from mtcnn import MTCNN
import cv2 as cv
import numpy as np

class DETECTION:
    def __init__(self):

        self.target_size = (160,160)
        self.detector = MTCNN()

    def extract_face(self, path):
        faces_result = []
        video = cv.VideoCapture(path)

        while video.isOpened() :
            ret, frame = video.read()
            if ret:
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)            
                faces = self.detector.detect_faces(rgb_frame)
                if len(faces) > 0:
                    for face in faces:
                        x, y, w, h = face['box']
                        face_image = rgb_frame[y:y+h, x:x+w]
                        face_image = cv.resize(face_image, self.target_size)

                        faces_result.append(np.asarray(face_image))
                else: continue
            else:
                break
        video.release()
        cv.destroyAllWindows()
        return np.asarray(faces_result)
    

