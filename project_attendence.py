import os
import cv2
import numpy as np
from PIL import Image

# faces=[]
# IDs=[]
face_detect=cv2.cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#
# recognizer=cv2.face.LBPHFaceRecognizer_create()
# path="/home/aparna/ml/training_data"
# subfolders=os.listdir(path)
#
# def ImageWithID(path):
#     for folders in subfolders:
#         folder_path=path+"/"+folders
#         images=os.listdir(folder_path)
#
#         for image_names in images:
#             image_paths=folder_path+"/"+image_names
#
#             face_Img=Image.open(image_paths).convert("L")
#             #facefirst=cv2.imread(image_path)
#             #ace_Img=cv2.cvtColor(facefirst,cv2.COLOR_BGR2GRAY)
#             faceNp=np.array(face_Img,'uint8')
#             ID1=image_paths.split('/')[-1]
#             ID=int(ID1.split("-")[0])
#             faces.append(faceNp)
#             IDs.append(ID)
#             cv2.imshow("training",faceNp)
#             cv2.waitKey(10)
#     return np.array(IDs),faces
#
# IDs,faces=ImageWithID(path)
# recognizer.train(faces,IDs)
# recognizer.write("recognizer.yml")
# cv2.destroyAllWindows

#face_detect
cam=cv2.VideoCapture(0)

rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("/home/aparna/ml/recognizer.yml")
ID=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL

while(True):

    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)
    #width=cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    #height=cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #video_frame=cv2.VideoWriter_fourcc(*'XVID')
    #video_output=cv2.VideoWriter('a.avi',video_frame,50.0,(int(width),int(height)))
    #face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    #x,frame=cam.read()
    #faces=face_cascade.detectMultiScale(frame,scaleFactor=1.05,minNeighbors=5)

    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        ID,conf=rec.predict(gray[y:y+h,x:x+w])
        if (ID==1):
         ID="aparna"
        if (ID==2):
         ID="avani"

        cv2.putText(img,str(ID),(w,y+h),font,2,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("cam",img)
    if (cv2.waitKey(1)==ord('q')):
        break;


cam.release
cv2.destroyAllWindows()
