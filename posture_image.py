import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
import setting
import glob
import time
import os
from cheatposture import preprocess
import settings
import cv2
import numpy as np
import os
import sqlite3
import smtplib
import emotions
import detector
#import settings
import smtplib 
#emotions.display()
settings.count=0
#facedetect = cv2.CascadeClassifier("haarcascade//haarcascade_frontalface_default.xml")
#
##cam = cv2.VideoCapture(0)
#recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read("/Users/sanjanasrinivasareddy/Desktop/nmit/Sitting-Posture-Recognition-master/haarcascade/recognizer/trainingdata.yml")
## id = 0
#def getProfile(id):
#    conn=sqlite3.connect("FaceBase.db")
#    # cmd="SELECT * FROM Peoples WHERE ID="+str(id)
#    cursor=conn.execute("SELECT * FROM Peoples WHERE id=?", (id,))
#    profile=None
#    for row in cursor:
#        profile=row
#    conn.close()
#    return profile
#from SimpleCV import *

#from SimpleCV import *
#from detector import path
#path='/Users/sanjanasrinivasareddy/Desktop/nmit/20200219_111952.jpeg'
tic=0
# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def process (input_image, params, model_params):
    ''' Start of finding the Key points of full body using Open Pose.'''
    oriImg = cv2.imread(input_image)  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    for m in range(1):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])
        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
        output_blobs = model.predict(input_img)
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = [] #To store all the key points which a re detected.
    peak_counter = 0
    
    prinfTick(1) #prints time required till now.

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    prinfTick(2) #prints time required till now.
    print()
    position = checkPosition(all_peaks) #check position of spine.
    eyesseen(all_peaks)
    #arm(all_peaks)
    #checkKneeling(all_peaks) #check whether kneeling oernot
    #checkHandFold(all_peaks) #check whether hands are folding or not.
    canvas1 = draw(input_image,all_peaks) #show the image.
    return canvas1 , position


def draw(input_image, all_peaks):
    canvas = cv2.imread(input_image)  # B,G,R order
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    return canvas


def checkPosition(all_peaks):
    try:
        f = 0
        if (all_peaks[16]):
            a = all_peaks[16][0][0:2] #Right Ear
            #print("hello")
            #print(a)
            f = 1
        else:
            a = all_peaks[17][0][0:2] #Left Ear
            #print("hi")
            #print(a)
        b = all_peaks[11][0][0:2] # Hip
        angle = calcAngle(a,b)
        degrees = round(math.degrees(angle))
        
        #print(b)
        #print(degrees)
        if (f):
            degrees = 180 - degrees
        if (degrees<70):
            return 1
        elif (degrees > 110):
            return -1
        else:
            return 0
    except Exception as e:
        print("person not in lateral view possibility of cheating")
        settings.count=settings.count+1
        if(settings.count>9):
            settings.q=settings.q+1
            print(preprocess(cv2.imread(settings.path+str(settings.i)+'.jpg')))
#            l=[]
#            g=0;
#
#            while(g<10):
#    
#                img = cv2.imread(settings.path+str(settings.i)+'.jpg')
#                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#                faces=facedetect.detectMultiScale(gray,1.3,5);
#                for(x,y,w,h) in faces:
#                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#                    id,conf=recognizer.predict(gray[y:y+h,x:x+w])
#                    profile=getProfile(id)
#                    #print(id)
#                    l.append(id)
#                g=g+1
#        #print(profile)
#        #if(profile != None):
#            #cv2.putText(img, "Name : "+str(profile[1]), (x,y+h+20),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
#            #cv2.putText(img, "Age : "+str(profile[2]), (x,y+h+45),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
#            #cv2.putText(img, "Gender : "+str(profile[3]), (x,y+h+70),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
#            #cv2.putText(img, output_text, (50,50),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
#
#    #cv2.imshow("Face",img);
#                    
#                    #if(cv2.waitKey(1)==ord('q')):
#                        #break;
#            print(l)
#            p=max(set(l), key = l.count)
#
#            profile=getProfile(p)
#            print(profile)
def eyesseen(all_peaks):
        
        f=0
        if(all_peaks[14]):
            a=all_peaks[14][0][0:2]
            f=f+1
        if(all_peaks[15]):
            b=all_peaks[15][0][0:2]
            f=f+1
        if(f==1):
            print("one eye is seen")
        if(f==2):
            print("not looking straight possibility of cheating")
            settings.count=settings.count+1
            if(settings.count>9):
                settings.q=settings.q+1
                print(preprocess(cv2.imread(settings.path+str(settings.i)+'.jpg')))
#                l=[]
#                g=0;
#
#                while(g<10):
#                    print('hello')
#                    img = cv2.imread(settings.path+str(settings.i)+'.jpg')
#                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#                    faces=facedetect.detectMultiScale(gray,1.3,5);
#                    for(x,y,w,h) in faces:
#                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#                        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
#                        profile=getProfile(id)
#                        print(id)
#                        l.append(id)
#                    g=g+1
#        #print(profile)
#        #if(profile != None):
#            #cv2.putText(img, "Name : "+str(profile[1]), (x,y+h+20),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
#            #cv2.putText(img, "Age : "+str(profile[2]), (x,y+h+45),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
#            #cv2.putText(img, "Gender : "+str(profile[3]), (x,y+h+70),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
#            #cv2.putText(img, output_text, (50,50),cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
#
#    #cv2.imshow("Face",img);
#                    
#                    #if(cv2.waitKey(1)==ord('q')):
#                        #break;
#                print(l)
#                
#                p=max(set(l), key = l.count)
#
#                profile=getProfile(p)
#                print(profile)
                
            #import detector
    #except Exception as e:
        #print("eye not found")
        
    

#calculate angle between two points with respect to x-axis (horizontal axis)
def calcAngle(a, b):
    try:
        ax, ay = a
        bx, by = b
        if (ax == bx):
            return 1.570796
        return math.atan2(by-ay, bx-ax)
    except Exception as e:
        print("unable to calculate angle")

        


def calcDistance(a,b): #calculate distance between two points.
    try:
        x1, y1 = a
        x2, y2 = b
        return math.hypot(x2 - x1, y2 - y1)
    except Exception as e:
        print("unable to calculate distance")




def showimage(img): #sometimes opencv will oversize the image when using using `cv2.imshow()`. This function solves that issue.
    screen_res = 1280, 720 #my screen resolution.
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def prinfTick(i): #Time calculation to keep a trackm of progress
    toc = time.time()
    #print ('processing time%d is %.5f' % (i,toc - tic))        

if __name__ == '__main__': #main function of the program
    tic = time.time()
    print('start processing...')
    extension = "*.jpg"
    #directory = os.path.join('/Users/sanjanasrinivasareddy/Desktop/realtimedata', extension)
    #files = sorted(glob.glob(directory))
    filelist = os.listdir(settings.path)
    #filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0]))

    
    
    model = get_testing_model()
    model.load_weights('./model/keras/model.h5')
    #for file in filelist:
    while(settings.i<len(filelist)-1):
        file=(settings.path + str(settings.i) + '.jpg')
        print(file)
        #new_img = Image(file)
        #new_img.show()
        #time.sleep(1) #wait for 1 second
        

        vi=False
        if(vi == False):
            
            params, model_params = config_reader()
            canvas, position= process(file, params, model_params)
            showimage(canvas)
            settings.i=settings.i+1
            if(settings.i%10==0):
                settings.count=0
    print('count of cheating is '+ str(settings.q))
    if(settings.q>6):
         
  
# creates SMTP session 
        s = smtplib.SMTP('smtp.gmail.com', 587) 
  
# start TLS for security 
        s.starttls() 
  
# Authentication 
        s.login("cheat.devp@gmail.com", "qwerty@123456") 
  
# message to be sent 
        m='student id: ' +str(settings.sad[0])+' name is :' +' '+settings.sad[1]+' age is : '+str(settings.sad[2])+' gender is: '+settings.sad[3]
        message = 'Count of Cheating is approximately '+str(settings.q)+' \nstudent who is cheating :\n '+m+ ' \nbehavior of student: '+settings.st1+' '+settings.st2
  
# sending the mail 
        s.sendmail("cheat.devp@gmail.com", "nishchaljs@gmail.com", message) 
  
# terminating the session 
        s.quit() 
        showimage(canvas)
            

       