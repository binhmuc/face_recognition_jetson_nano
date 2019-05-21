import dlib
import cv2
from keras.models import load_model
from keras.utils import CustomObjectScope
import numpy as np
import pickle
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import sys
with open("./classifier.pkl", 'rb') as f:
    if sys.version_info[0] < 3:
            (le, clf) = pickle.load(f)
    else:
            (le, clf) = pickle.load(f, encoding='latin1')

with CustomObjectScope({'tf': tf}):
    model = load_model('./model/nn4.small2.v1.h5')

detector = dlib.get_frontal_face_detector()
rgb = cv2.VideoCapture(0)

labels = ["ha", "lai", "quynh", "thien", "thin", "xupe"]

def get128keypoint(img):
    img = img[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    y1 = model.predict_on_batch(x_train)
    return y1[0]

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


while True:
    _, img = rgb.read()
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # rects = detector(gray, 1)

    # for rect in rects:
    #     (x,y,w,h) = rect_to_bb(rect)
    #     cv2.rectangle(resized, (int(x), int(y )), (int(x + w), int(y + h)), (0, 255, 0), 3)
    # cv2.imshow("Resized image", resized)

    rects = detector(gray, 1)
    for rect in rects:
        (x,y,w,h) = rect_to_bb(rect)
        face = cv2.resize(resized[y:y+h, x:x+w], (96,96))


        rep = get128keypoint(face).reshape(1, -1)
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        print labels[maxI]

        cv2.rectangle(resized, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 3)
        cv2.putText(resized, labels[maxI], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imshow("Resized image", resized)
    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()

