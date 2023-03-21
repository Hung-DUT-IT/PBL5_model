from mtcnn import MTCNN
import cv2 as cv
import os
class DETECTION:
    def __init__(self):

        self.target_size = (160,160)
        self.detector = MTCNN()

    def extract_face(self, path):
        faces_result = []
        video = cv.VideoCapture(path)

        count = 0
        while video.isOpened() and count < 20:
            ret, frame = video.read()
            if ret:
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)            
                faces = self.detector.detect_faces(rgb_frame)
                if len(faces) > 0:
                    for face in faces:
                        x, y, w, h = face['box']
                        face_image = rgb_frame[y:y+h, x:x+w]
                        face_image = cv.resize(face_image, self.target_size)

                        faces_result.append(face_image)
                    count += 1
                else: continue
            else:
                break
        video.release()
        cv.destroyAllWindows()
        return faces_result
    

