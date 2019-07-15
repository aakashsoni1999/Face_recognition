import face_model
import argparse
import sys
import pickle
import os
import cv2
import numpy as np
import datetime
import glob
import sys
import imutils

sys.path.append("../RetinaFace/")

from retinaface import RetinaFace


count_for_face_detection = 1

gpuid = 0
detector = RetinaFace('../models/R50', 0, gpuid, 'net3')

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model,0',
                    help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int,
                    help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int,
                    help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24,
                    type=float, help='ver dist threshold')
args = parser.parse_args()
model = face_model.FaceModel(args)
dim = (112, 112)


def detect_faces(img, scales, thresh):
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
    flip = False
    for c in range(count_for_face_detection):
        faces, landmarks = detector.detect(
            img, thresh, scales=scales, do_flip=flip)
    face_locations = []
    if faces is not None:
        print('find', faces.shape[0], 'faces')
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            color = (0, 0, 255)
            face_locations.append((box[1], box[2], box[3], box[0]))
    return face_locations


dataset_path = './Training_data/'
for j in os.listdir(dataset_path):
    if os.path.isdir(dataset_path + j):
        curr = dataset_path + j
        known_face_encodings = []
        known_face_names = []
        flag = 0
        print(curr)
        for k in os.listdir(curr):
            if k.endswith('face_names.data'):
                with open(curr + "/face_names.data", "rb") as f:
                    temp = []
                    name = " "
                    temp = pickle.load(f)
                    for m in temp:
                        name = m
                        name = name[:7]
                    if(name == "Unknown"):
                        flag = -1
                        break
                flag = 1
                break
        if ((flag == 0)or(flag == -1)):
            for k in os.listdir(curr):
                if k.endswith('.jpg'):
                    try:
                        if (flag == -1):
                            known_face_names.append(j)
                        else:
                            image = cv2.imread(dataset_path + j + '/' + k)
                            thresh = 0.8
                            scales = [1024, 1980]
                            face_locations = detect_faces(
                                image, scales, thresh)
                            for (top, right, bottom, left) in face_locations:
                                image = image[top:bottom, left:right]
                                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                                image = model.get_input(image)
                                image_encoding = model.get_feature(image)
                                print("Successful Capture")
                            if(len(image_encoding) > 0):
                                known_face_encodings.append(image_encoding)
                                known_face_names.append(j)
                    except Exception as error:
                        print("Deleting Image " + k)
                        print(error)
                if k.endswith('.mp4'):
                    video_capture = cv2.VideoCapture(curr + "/" + str(k))
                    while True:
                        ret, frame = video_capture.read()
                        if (ret == False):
                            break
                        try:
                            frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
                            frame = imutils.rotate(frame, -90)
                            thresh = 0.8
                            scales = [1024, 1980]
                            face_locations = detect_faces(
                                frame, scales, thresh)
                            for (top, right, bottom, left) in face_locations:
                                frame = frame[top:bottom, left:right]
                                frame = model.get_input(frame)
                                image_encoding = model.get_feature(frame)
                            if(len(image_encoding) > 0):
                                known_face_encodings.append(image_encoding)
                                known_face_names.append(j)
                                print("successful capture")
                        except Exception as error:
                            print(error)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    video_capture.release()
                    cv2.destroyAllWindows()
            if(flag == 0):
                with open(curr + "/" + "face_encodings.data", "wb") as f:
                    pickle.dump(known_face_encodings, f)
            with open(curr + "/" + "face_names.data", "wb") as f:
                pickle.dump(known_face_names, f)
