# Face Recognition

# Importing the libraries
import cv2
import numpy as np
import math
import threading

mindif=0.08
can=False
prnt=[]
# Loading the cascades
eye_cascade = cv2.CascadeClassifier('C:/xampp/htdocs/anaconda/Computer_Vision_A_Z_Template_Folder/Module_1_Face_Recognition/haarcascade_eye.xml')
np.seterr(all='warn')

def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t

def cantrue():
    global prnt
    global diseses
    prnt=[]
    for m in range(len(diseses)):
        prnt.append(diseses[m])
    
set_interval(cantrue,1)
#Mask an image so all pixels outside feature circle are black
def mask_image_by_feature(image, feature):
    circle_mask_image = np.zeros(image.shape, dtype=np.uint8)
    cv2.circle(circle_mask_image, (int(feature.pt[0]), int(feature.pt[1])), int(feature.size/2), 1, -1)
    masked_image = (image * circle_mask_image).astype(np.uint8)
    return masked_image

#Find average brightness of pixels under a feature's circle
def find_average_brightness_of_feature(image, feature):
    feature_image = mask_image_by_feature(image, feature)
    total_value = feature_image.sum()
    area = np.pi * ((feature.size/2)**2)
    return total_value/area

def sort_features_by_brightness(image, features):
    features_and_brightnesses = [(find_average_brightness_of_feature(image, feature), feature) for feature in features]
    features_and_brightnesses.sort(key = lambda x:x[0])
    return [fb[1] for fb in features_and_brightnesses]

def find_pupil(gray_image,frame, minsize=.1, maxsize=.5):
    detector = cv2.MSER_create()
    features_all = detector.detect(gray_image)
    features_big = [feature for feature in features_all if feature.size > gray_image.shape[0]*minsize]
    features_small = [feature for feature in features_big if feature.size < gray_image.shape[0]*maxsize]
    if len(features_small) == 0:
        return None
    features_sorted = sort_features_by_brightness(gray_image, features_small)
    pupil = features_sorted[0]
    return (int(pupil.pt[0]), int(pupil.pt[1]), int(pupil.size/2))

def separate(x1, x2, y1, y2):
    return math.sqrt( ((int(x2) - int(x1))**2 + (int(y2) - int(y1))**2) )
# Defining a function that will do the detections

def detect(gray, frame):
    global can
    global prnt
    global diseses
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 30)
    diseses= [
        {"min": 0,   "max": 25,  "part": "Lungs",    "positive": 0, "all": 1},
        {"min": 25,  "max": 55,  "part": "Ears",     "positive": 0, "all": 1},
        {"min": 55,  "max": 125, "part": "Brains",   "positive": 0, "all": 1},
        {"min": 125, "max": 180, "part": "Mouth",    "positive": 0, "all": 1},
        {"min": 180, "max": 230, "part": "Back",     "positive": 0, "all": 1},
        {"min": 230, "max": 265, "part": "Abdominal","positive": 0, "all": 1},
        {"min": 265, "max": 310, "part": "Leg",      "positive": 0, "all": 1},
        {"min": 310, "max": 325, "part": "Hips",     "positive": 0, "all": 1},
        {"min": 325, "max": 340, "part": "Arms",     "positive": 0, "all": 1},
        {"min": 340, "max": 360, "part": "Chest",    "positive": 0, "all": 1}
    ]
    for (x, y, w, h) in eyes:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        p=find_pupil(roi_gray,roi_color, 0.03 , 0.38)
        
        if(p!=None):
            if(0.3<(p[0]/w) and (p[0]/w) <0.7 and 0.4<(p[1]/h) and (p[1]/h)<0.6):
                radius=int(h*0.17)
                cv2.circle(roi_color, (p[0], p[1]), p[2], (0, 255, 255), 1)
                cv2.circle(roi_color, (p[0], p[1]), radius, (0, 255, 255), 1)
                sump=0
                count=0
                
                for x in range(p[0]-radius+1, p[0]+radius-1):
                    for y in range(p[1]-radius+1, p[1]+radius-1):
                        if(separate(p[0], x, p[1], y)<radius and separate(p[0], x, p[1], y)>p[2]):
                            sump+=sum(roi_color[x, y])
                            count+=1
                if(count>0):
                    average=sump/count/3
                    for x in range(p[0]-radius+1, p[0]+radius-1):
                        for y in range(p[1]-radius+1, p[1]+radius-1):
                            if(separate(p[0], x, p[1], y)<radius and separate(p[0], x, p[1], y)>p[2]):
                                diff=average/sum(roi_color[x, y])-1
                                angle=(math.atan2(-y+p[1], x-p[0])/math.pi+1)*180
                                positive=False
                                if(diff>mindif):
                                    cv2.circle(roi_color,(x,y),1,(255,0,255),2)
                                    positive=True
                                for disese in diseses:
                                    if(angle>=disese["min"] and angle<disese["max"]):
                                        disese["all"]+=1
                                        if(positive):
                                            disese["positive"]+=1
                    
    font = cv2.FONT_HERSHEY_SIMPLEX
    counter=0
    for dis in prnt:
        if(dis["all"]>0):
            counter+=1
            cv2.putText(frame,dis["part"]+": " + str(int(dis["positive"]/(dis["all"])*100)) + "%",(10,counter*25), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        video_capture.release()
        cv2.destroyAllWindows()
        break
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
video_capture.release()
cv2.destroyAllWindows()