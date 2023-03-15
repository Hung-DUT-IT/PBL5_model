from mtcnn import MTCNN
import cv2 as cv

class DETECTION:
    def __init__(self):

        self.target_size = (160,160)
        self.face = []
        self.label_ids = {}
        self.current_id = 0
        self.detector = MTCNN()

    def extract_face(self, file_name):

        img = cv.imread(file_name)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        x, y, width, height = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img [y:y+ height, x:x+ width]
        face = cv.resize(face, self.target_size)
        return face