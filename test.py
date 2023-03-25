import cv2 as cv 
from mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt

from Preprocessing.Extract_embeddings import EXTRACT_EMBEDDING
detector = MTCNN()
embedder = EXTRACT_EMBEDDING()


file_name = "C:/Users/huuhu/Learning/Data_Analytics\PBL\PBL5_model\DataSet\processed\AnhSon/AnhSon0.jpg"
img = cv.imread(file_name)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

x, y, width, height = detector.detect_faces(img)[0]['box']
x,y = abs(x), abs(y)
face = img [y:y+ height, x:x+ width]
face = cv.resize(face, (160, 160))
print(face.shape)



yhat= embedder.get_embedding(face)

print(yhat.shape)
plt.plot(yhat) 
plt.show()