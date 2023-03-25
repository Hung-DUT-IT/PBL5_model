from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Preprocessing.Load_data import LOAD_DATA
from Preprocessing.Extract_embeddings import EXTRACT_EMBEDDING
from mtcnn import MTCNN
from sklearn.svm import SVC
import cv2 as cv
from sklearn.preprocessing import LabelEncoder




build_dataset = LOAD_DATA()
detector = MTCNN()
embedding = EXTRACT_EMBEDDING()
encoder = LabelEncoder()

X, Y  = build_dataset.load_faces()



encoder.fit(Y)
Y = encoder.transform(Y)




EMBEDDED_X = embedding.get_embeddings(X)

X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, train_size=0.8, random_state=17)

# Create and train the SVC classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train,Y_train)


ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)


print(accuracy_score(Y_train, ypreds_train))
print(accuracy_score(Y_test, ypreds_test))

cap = cv.VideoCapture(0)

while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces in the frame using MTCNN
    faces = detector.detect_faces(frame)
    
    # Loop through each detected face
    for face in faces:
        # Crop the face from the frame
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)
        test_image = frame[y:y+h, x:x+w]
        
        # Preprocess the face image for FaceNet
        test_image = cv.cvtColor(test_image, cv.COLOR_BGR2RGB)
        test_image = cv.resize(test_image, (160, 160))

        test_image_enc = embedding.get_embedding(test_image)
        
        face_name = model.predict([test_image_enc])
        final_name = encoder.inverse_transform(face_name)[0]


        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
        cv.putText(frame, str(final_name), (x+10,y-10), cv.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 3, cv.LINE_AA)

    
    # Display the frame with the detected faces
    cv.imshow('Frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv.destroyAllWindows()