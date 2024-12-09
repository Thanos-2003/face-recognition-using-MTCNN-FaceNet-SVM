import cv2
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
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

grp_img =   cv2.imread("C:/Users/Thanos/Downloads/20241206_092332.jpg")


rgb_img = cv2.cvtColor(grp_img,cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(grp_img,cv2.COLOR_BGR2GRAY)
faces = haarcascades.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
for x, y, w, h in faces:
    img = grp_img[y:y+h, x:x+w]
    img = cv2.resize(img, (160, 160))
    img = np.expand_dims(img, axis=0)
    y_pred=facenet.embeddings(img)
    face_name=model.predict(y_pred)
    final_name = encoder.inverse_transform(face_name)[0]

    cv2.rectangle(grp_img,(x,y),(x+w,y+h),(255,0,0),5)
    cv2.putText(grp_img,str(final_name),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)


grp_img = cv2.resize(grp_img, (1920,1080))
cv2.imshow("face recognition: ", grp_img)
cv2.waitKey(0)
