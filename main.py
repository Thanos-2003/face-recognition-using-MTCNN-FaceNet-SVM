import cv2
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
facenet = FaceNet()
faces_embeddings = np.load('face_embeddings_don_4classes.npz')
Y= faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

haarcascades = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model= pickle.load(open("svm_model_160x160.pkl",'rb'))

cap = cv2.VideoCapture(0)


while cap.isOpened():
    _,frame = cap.read()
    rgb_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = haarcascades.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    for x, y, w, h in faces:
        img = frame[y:y+h, x:x+w]
        img = cv2.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)
        y_pred=facenet.embeddings(img)
        face_name=model.predict(y_pred)
        final_name = encoder.inverse_transform(face_name)[0]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),10)
        cv2.putText(frame,str(final_name),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)

    cv.imshow("face recognition: ", frame)
    if cv2.waitKey(1) & ord('q')==27:
        break

cap.release()
cv2.destroyAllWindows()