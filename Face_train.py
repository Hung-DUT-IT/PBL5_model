import cv2 as cv 
from mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet
import matplotlib.pyplot as plt
detector = MTCNN()
embedder = FaceNet()
file_name = "C:/Users/huuhu/Data_Analytics/PBL/PBL5_model/DataSet/raw/TranThanh/img_6.jpg"
img = cv.imread(file_name)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

x, y, width, height = detector.detect_faces(img)[0]['box']
x,y = abs(x), abs(y)
face = img [y:y+ height, x:x+ width]
face = cv.resize(face, (160, 160))


face_img = face.astype('float32') # 3D(160x160x3)
face_img = np.expand_dims(face_img, axis=0) 
# 4D (Nonex160x160x3)
yhat= embedder.embeddings(face_img)
plt.plot(yhat[0]) 
plt.show()
