import face_recognition
import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import xlwt
from xlwt import Workbook

wb=Workbook()

sheet=wb.add_sheet('Log Record')

sheet.write(0,0,'Sno.')
sheet.write(0,1,'Name')
sheet.write(0,2,'Status')
sheet.write(0,3,'Time')
sheet.write(0,4,'Date')
sheet.write(0,5,'Total_time')
sheet.write(0,6,'Hours')
sheet.write(0,7,'Minutes')
sheet.write(0,8,'Seconds')

previous=[]
index=1
s_no=1
persons_time={}
f_r=1

known_face_encodings = []
known_face_names = []

with open("face_encodings.data","rb") as f:
                known_face_encodings=pickle.load(f)
with open("face_names.data","rb") as f:
                known_face_names=pickle.load(f)
            
video_capture = cv2.VideoCapture(0)
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
count=0

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=1/f_r, fy=1/f_r)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
                    
        current=[]
        for (top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):
                    name = "Unknown"
                    if (not(len(known_face_encodings)==0)):
                                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                    best_match_index = np.argmin(face_distances)
                                    if matches[best_match_index]:
                                                    name = known_face_names[best_match_index]
                                                    current.append(name)
                    if (name=="Unknown"):
                                known_face_encodings.append(face_encoding)
                                name="Unknown"+str(index)
                                known_face_names.append(name)
                                per=0.25
                                width=(right-left)*f_r
                                height=(bottom-top)*f_r
                                x1=int(left*f_r-(width)*per)
                                y1=int(top*f_r-(height)*per)
                                y2=int(bottom*f_r+(height)*per)
                                x2=int(right*f_r+(width)*per)
                                unknown_image=frame[y1:y2,x1:x2]
                                cv2.imwrite("Unknown"+str(index)+".jpg",unknown_image)
                                index=index+1
                    face_names.append(name)
        
        for i in current:
                flag=0
                for j in previous:
                        if(i==j):
                                flag=1
                                break
                if(flag==0):
                        sheet.write(s_no,0,str(s_no))
                        sheet.write(s_no,1,str(i))
                        sheet.write(s_no,2,'ENTRY')
                        sheet.write(s_no,3,str(datetime.now().time()))
                        sheet.write(s_no,4,str(datetime.now().date()))
                        if str(i) in persons_time:
                                        time_diff=datetime.now()-persons_time[str(i)]
                                        t_seconds=time_diff.total_seconds()
                                        hours=t_seconds//3600
                                        sheet.write(s_no,6,str(hours))
                                        t_seconds=t_seconds%3600
                                        minutes=t_seconds//60
                                        sheet.write(s_no,7,str(minutes))
                                        t_seconds=t_seconds%60
                                        seconds=t_seconds
                                        sheet.write(s_no,8,str(seconds))
                        s_no=s_no+1
                        
        for i in previous:
                flag=0
                for j in current:
                        if(i==j):
                                flag=1
                                break
                if(flag==0):
                        sheet.write(s_no,0,str(s_no))
                        sheet.write(s_no,1,str(i))
                        sheet.write(s_no,2,'EXIT')
                        sheet.write(s_no,3,str(datetime.now().time()))
                        sheet.write(s_no,4,str(datetime.now().date()))
                        if str(i) in persons_time:
                                    del persons_time[str(i)]
                        else:
                                    persons_time[str(i)]=datetime.now()
                        s_no=s_no+1
                        
        previous=list(current)
            
         
    if(count%3==0):
            process_this_frame = not process_this_frame
    count=count+1

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= f_r
        right *= f_r
        bottom *= f_r
        left *= f_r
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0),2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX,1.0,(0,255,255),2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

wb.save('Log record.ods')

video_capture.release()
cv2.destroyAllWindows()
