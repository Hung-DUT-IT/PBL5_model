from keras_facenet import FaceNet
import numpy as np

class EXTRACT_EMBEDDING:
    def __init__(self):
        self.model = FaceNet()
        self.EMBEDDED_X = []

    def get_embedding(self, face_img):

        face_img = face_img.astype('float32') # 3D(160x160x3)
        
        face_img  = np.expand_dims(face_img, axis=0) # 4D (Nonex160x160x3)

        Embedded = self.model.embeddings(face_img)
        return Embedded[0] # 512D image (1x1x512)

    def get_embeddings(self, faces):
        for face in faces:
            self.EMBEDDED_X.append(self.get_embedding(face))

        return np.asarray(self.EMBEDDED_X)