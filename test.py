import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from Resnet import InceptionResNetV2
from sklearn.preprocessing import Normalizer
from keras.models import load_model


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

face_encoder = InceptionResNetV2()
path_m = "C:/Users/huuhu/Learning/Data_Analytics/PBL/PBL5_model/facenet_keras_weights.h5"
face_encoder.load_weights(path_m)

file = "C:/Users/huuhu/Learning/Data_Analytics/PBL/PBL5_model/DataSet/processed/AnhDung/AnhDung2.jpg"
img_1 = cv.imread(file)
img_1 = cv.cvtColor(img_1,  cv.COLOR_BGR2RGB)
print(img_1.shape)
face_1 = normalize(img_1)
face_d_1 = np.expand_dims(face_1, axis=0)
encode = face_encoder.predict(face_d_1)[0]
plt.plot(encode)



file_name = "C:/Users/huuhu/Learning/Data_Analytics/PBL/PBL5_model/DataSet/processed/AnhDung/AnhDung220.jpg"
img = cv.imread(file_name)
img = cv.cvtColor(img,  cv.COLOR_BGR2RGB)

face = normalize(img)
face = cv.resize(face, (160,160))
face_d = np.expand_dims(face, axis=0)
db_encode = face_encoder.predict(face_d)[0]
plt.plot(db_encode)

plt.show()


from scipy.spatial.distance import cosine
distance = float("inf")

dist = cosine(db_encode, encode)
print(dist<0.5)