from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import numpy as np
import pickle
with CustomObjectScope({'tf': tf}):
    model = load_model('./model/nn4.small2.v1.h5')


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=192)

def getImageAlgin(imgPath):
    image = cv2.imread(imgPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) > 0:
        rect = faces[0]
        faceAligned = fa.align(image, gray, rect)
        rects = detector(faceAligned, 1)
        if len(rects) > 0:
            rect = detector(faceAligned, 1)[0]
            (x, y, w, h) = rect_to_bb(rect)
            return cv2.resize(faceAligned[y:y + h, x:x + w], (96,96))
        else :
            return False
    else :
        return False



def get128keypoint(imgPath):
    faceAligned = getImageAlgin(imgPath)
    if faceAligned is not False:
        img = faceAligned[...,::-1]
        img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
        x_train = np.array([img])
        y1 = model.predict_on_batch(x_train)
        return y1[0]
    else :
        return False

data = []
label = []
listfolder = glob.glob("./images/train/*")
for i, value in enumerate(listfolder):
    listfile = glob.glob(value + "/*.*")
    for img in listfile:
        print(img)
        data.append(get128keypoint(img))
        label.append(i)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(data, label)

labels = ["ha", "lai", "quynh", "thien", "thin", "xupe"]
le = LabelEncoder().fit(labels)
print(le)
fName = "classifier.pkl"
with open(fName, 'wb') as f:
    pickle.dump((le, clf), f)


rep = get128keypoint('./images/mm6.jpg').reshape(1, -1)
predictions = clf.predict_proba(rep).ravel()
maxI = np.argmax(predictions)
confidence = predictions[maxI]
print(predictions, confidence)