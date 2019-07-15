import cv2
import numpy as np
import os
import pickle
import datetime
from datetime import datetime
import dateutil.parser
import face_model
import sys
import argparse
import os
import glob
import sys
import imutils
sys.path.append("../RetinaFace/")
from retinaface import RetinaFace
from excel_write import write_in_file
from excel_write import final_write
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db = client['jbmDB']

faceFrame = db.faceFrame

os.system("python3 Face_trainer.py")

person_details = {}

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

previous = []
s_no = 1
persons_time = {}
total_time = {}
margin = 0
dimension = (112, 112)
face_locations = []
face_encodings = []
process_this_frame = True
count_of_frames = 0
known_face_encodings=[]
known_face_names=[]
dim=(112,112)

for j in os.listdir('./Training_data/'):
    if os.path.isdir('./Training_data/' + j):
        for k in os.listdir('./Training_data/' + j):
            if k.endswith('face_encodings.data'):
                with open("./Training_data/" + j + "/face_encodings.data", "rb") as f:
                    temp = []
                    temp = pickle.load(f)
                    for i in temp:
                        known_face_encodings.append(i)
            if k.endswith('face_names.data'):
                with open("./Training_data/" + j + "/face_names.data", "rb") as f:
                    temp = []
                    temp = pickle.load(f)
                    for i in temp:
                        known_face_names.append(i)
print(len(known_face_encodings))

# video_capture = cv2.VideoCapture("rtsp://admin:password%40123@192.1.58.18:554/live/0/MAIN"
# video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture("d.mp4")
# video_capture = cv2.VideoCapture("yy.mov")
# video_capture = cv2.VideoCapture("rtsp://admin:password123@192.1.10.76:554")
# video_capture = cv2.VideoCapture("rtsp://admin:password%40123@192.1.58.8:554/live/0/MAIN")
#video_capture = cv2.VideoCapture("rtsp://admin:password123@192.1.10.76:554")

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
    faces = None
    for c in range(count_for_face_detection):
        faces, landmarks = detector.detect(
            img, thresh, scales=scales, do_flip=flip)
    face_locations = []
    if faces is not None:
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            color = (0, 0, 255)
            face_locations.append((box[1], box[2], box[3], box[0]))
    return face_locations

while True:
    if (os.listdir("./samplevideos")==[]):
        break
    src=sorted(glob.glob("./samplevideos/*"))[0]
    video_capture=cv2.VideoCapture(src)
    while True:
        if (not os.path.exists('./saved_frames/'+str(datetime.now().date()))):
            os.mkdir('./saved_frames/'+str(datetime.now().date()))
        thresh = 0.8
        scales = [1024, 1980]
        ret, frame = video_capture.read()
        if (ret == True):
            small_frame=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        else:
            break
        if process_this_frame:
            face_locations = detect_faces(small_frame, scales, thresh)
            face_names = []
            current = []
            persons_in_one_frame = []
            person_details = {}
            dateTimeNow = datetime.now()
            person_details["time"]= dateutil.parser.parse(dateTimeNow.isoformat())
            person_details["personsDetected"] = []
            for (top, right, bottom, left) in face_locations:
                name = "Unknown"
                image = small_frame[top:bottom, left:right]
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                try:
                    image = model.get_input(image)
                    image_encoding = model.get_feature(image)
                    face_encoding = image_encoding
                except Exception as error:
                    continue
                score=0
                if (not(len(known_face_encodings) == 0)):
                    face_distances = []
                    for c in known_face_encodings:
                        sim = np.dot(image_encoding, c.T)
                        face_distances.append(sim)
                    best_match_index = np.argmax(face_distances)
                    if (face_distances[best_match_index] >= 0.45):
                        name = known_face_names[best_match_index]
                        score=face_distances[best_match_index]
                        current.append(name)


                    person_details["personsDetected"].append({
                        "bbox" : [top.item(), right.item(), bottom.item(), left.item()],
                        "empId" : name
                    })
                    cv2.rectangle(small_frame, (left, top),(right, bottom), (0, 255, 0), 2)
                    cv2.putText(small_frame, name+" "+str("{0:.2f}".format(score*100)), (left + 6, bottom - 6),cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
                    if (not os.path.exists('./saved_frames/'+str(dateTimeNow.date())+"/"+str('cam01'))):
                        os.mkdir('./saved_frames/'+str(dateTimeNow.date())+'/'+str('cam01'))
                    cv2.imwrite('./saved_frames/'+str(dateTimeNow.date())+'/'+str('cam01')+'/'+str(dateTimeNow.time())+'.jpg',small_frame)

                    person_details["framePath"] = './saved_frames/'+str(dateTimeNow.date())+'/'+str('cam01')+'/'+str(dateTimeNow.time())+'.jpg'
            faceFrame.insert_one(person_details)
        count_of_frames=count_of_frames+1
        if(count_of_frames%1==0):
            process_this_frame= process_this_frame
        s_no=write_in_file(current,previous,total_time,persons_time,s_no)
        previous = list(current)
        cv2.imshow('Video', small_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    os.remove(src)
    video_capture.release()

final_write(total_time,s_no)
