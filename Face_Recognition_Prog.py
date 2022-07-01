import cv2
import os
from cv2 import imshow
import numpy as np
import face_recognition

def findEncode(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


path ="Face_recog_images"
images = []
className = []
mylist = os.listdir(path)
print (mylist)

for cls in mylist:
    currentImg=face_recognition.load_image_file(f"{path}/{cls}")
    images.append(currentImg)
    className.append(os.path.splitext(cls)[0])

print(className)


knownEncodes=findEncode(images)
# print(len(knownEncodes))

cap = cv2.VideoCapture("Images/RPReplay_Final1656631051.mov")

while True:
    ret,frame=cap.read()
    small_img=cv2.resize(frame,(0,0),None,0.25,0.25)
    small_img=cv2.cvtColor(small_img,cv2.COLOR_BGR2RGB)
    faces=face_recognition.face_locations(small_img)
    encodeCur=face_recognition.face_encodings(small_img,faces)
    # image_encoding = face_recognition.face_encodings(images)

    for encodeface,floc in zip(encodeCur,faces):
        matches=face_recognition.compare_faces(knownEncodes,encodeface)
        fdis=face_recognition.face_distance(knownEncodes,encodeface)
        print(fdis)
        mIndex=np.argmin(fdis)

        if matches[mIndex]:
            name=className[mIndex].upper()
            print (name)
            y1,x1,y2,x2=floc
            y1,x1,y2,x2=y1*4,x1*4,y2*4,x2*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-32),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x2+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    
    cv2,imshow("Webcam",frame)
    if cv2.waitKey(1)==ord('q'):
        break


