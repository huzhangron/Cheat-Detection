import cv2
import numpy as np
import os
import sqlite3
import settings
import smtplib 
#path='/Users/sanjanasrinivasareddy/Desktop/nmit/20200219_111952.jpeg'
facedetect = cv2.CascadeClassifier("haarcascade//haarcascade_frontalface_default.xml")

#cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/Users/sanjanasrinivasareddy/Desktop/nmit/Sitting-Posture-Recognition-master/haarcascade/recognizer/trainingdata.yml")
# id = 0
# font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,1,1,0,1)

def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    # cmd="SELECT * FROM Peoples WHERE ID="+str(id)
    cursor=conn.execute("SELECT * FROM Peoples WHERE id=?", (id,))
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile
l=[]
i=0
   
filelist = os.listdir(settings.path)
try:
    xx=0
    while(xx<len(filelist)):
        file=(settings.path + str(xx) + '.jpg')
        print(file)
    
        img = cv2.imread(file)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=facedetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            id,conf=recognizer.predict(gray[y:y+h,x:x+w])
            if(id==120):
                id=4
            profile=getProfile(id)
            print(id)
            l.append(id)
        xx=xx+1
except:
    xx=xx+1
    
        #print(profile)
        #if(profile != None):
            #cv2.putText(img, "Name : "+str(profile[1]), (x,y+h+20),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            #cv2.putText(img, "Age : "+str(profile[2]), (x,y+h+45),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            #cv2.putText(img, "Gender : "+str(profile[3]), (x,y+h+70),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            #cv2.putText(img, output_text, (50,50),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)

    #cv2.imshow("Face",img);
    #
print(l)
p=max(set(l), key = l.count)

profile=getProfile(p)
settings.sad=profile
print('student profile is')
print(profile)
#s = smtplib.SMTP('smtp.gmail.com', 587) 
  

#s.starttls() 
  

#s.login("cheat.devp@gmail.com", "qwerty@123456") 
#str1=" " 

#m=str(profile[0])+' '+profile[1]+' '+str(profile[2])+' '+profile[3]
#print(m)

#s.sendmail("cheat.devp@gmail.com", "nishchaljs@gmail.com", m) 
  

#s.quit() 

