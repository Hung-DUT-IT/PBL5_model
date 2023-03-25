from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Preprocessing.Load_data import LOAD_DATA
from Preprocessing.Extract_embeddings import EXTRACT_EMBEDDING
import numpy as np
from sklearn.svm import SVC
import pickle


class Model:
    def __init__(self):
        self.load_data = LOAD_DATA()
        self.extract_embedding = EXTRACT_EMBEDDING()

        self.X = []
        self.Y = []
        self.EMBEDDED_X = []
        self.model = SVC(kernel='linear', probability=True)


    def load_mode(self):
        return 

    def save_embedding(self):
        self.X, self.Y = self.load_data.load_faces()
        self.EMBEDDED_X = self.extract_embedding.get_embeddings(self.X)
        np.savez_compressed('faces_embeddings_done.npz', self.EMBEDDED_X, self.Y)

    def trainning(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.EMBEDDED_X, self.Y, shuffle=True, random_state=17, stratify=self.Y)

        self.model.fit(X_train, Y_train)

        ypreds_train = self.model.predict(X_train)
        ypreds_test = self.model.predict(X_test)
        accuracy_score(Y_train, ypreds_train)
        
        return accuracy_score(Y_test,ypreds_test)

    def save_model(self):
        with open('svm_model_160x160.pkl', 'wb') as f:
            pickle.dump(self.model, f)

