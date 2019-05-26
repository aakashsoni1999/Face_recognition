import face_recognition
import pickle
import os

known_face_encodings = []
known_face_names = []
dataset_path='.//'
for j in os.listdir(dataset_path):
                    if j.endswith('.jpg'):
                            image=face_recognition.load_image_file(j)
                            image_encoding=face_recognition.face_encodings(image)[0]
                            known_face_encodings.append(image_encoding)
                            known_face_names.append(j[:-4])
                            
with open("face_encodings.data","wb") as f:
                pickle.dump(known_face_encodings,f)

with open("face_names.data","wb") as f:
                pickle.dump(known_face_names,f)
