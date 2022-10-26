import face_recognition
import cv2
import numpy as np
import csv 
from datetime import datetime

video_capture = cv2.VideoCapture(0)

justin_bieber_image = face_recognition.load_image_file("justin_bieber.jfif")
justin_bieber_encoding = face_recognition.face_encodings(justin_bieber_image)[0]

ratan_tata_image = face_recognition.load_image_file("ratan_tata.jpg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

central_cee_image = face_recognition.load_image_file("central_cee.jpg")
central_cee_encoding = face_recognition.face_encodings(central_cee_image)[0]

chhatrapati_shivaji_maharaj_image = face_recognition.load_image_file("chhatrapati_shivaji_maharaj.jpg")
chhatrapati_shivaji_maharaj_encoding = face_recognition.face_encodings(chhatrapati_shivaji_maharaj_image)[0]

christiano_ronaldo_image = face_recognition.load_image_file("christiano_ronaldo.jfif")
christiano_ronaldo_encoding = face_recognition.face_encodings(christiano_ronaldo_image)[0]

MS_dhoni_image = face_recognition.load_image_file("MS_dhoni.jfif")
MS_dhoni_encoding = face_recognition.face_encodings(MS_dhoni_image)[0]

known_face_encoding = [
    justin_bieber_encoding,
    ratan_tata_encoding,
    central_cee_encoding,
    chhatrapati_shivaji_maharaj_encoding,
    christiano_ronaldo_encoding,
    MS_dhoni_encoding
]

known_face_names = [
    "justin_bieber",
    "ratan_tata",
    "central_cee",
    "chhatrapati_shivaji_maharaj",
    "christiano_ronaldo",
    "MS_dhoni"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = [] 
s=True


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


f = open(current_date+'.excel workbook','w+',newline='')
lnwriter =  csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])   


        cv2.imshow("attendence system",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


video_capture.release()
cv2.destroyAllWindows()
f.close()







