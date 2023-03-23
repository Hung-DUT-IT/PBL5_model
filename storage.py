import os
from sklearn.model_selection import train_test_split
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
processed_folder = os.path.join(BASE_DIR, "DataSet/processed") 
folders = os.listdir(processed_folder)
X = []
y = []
for folder in folders:
    folder_path = os.path.join(processed_folder, folder)
    images = os.listdir(folder_path)
    for image in images:
        image_path = os.path.join(folder_path, image)
        X.append(image_path)
        y.append(folder)
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
