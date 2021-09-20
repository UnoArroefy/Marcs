import cv2
import numpy as np
from keras.models import load_model

val = None
Ds = None
model = load_model('mask/model8.h5')

face_clsfr=cv2.CascadeClassifier('mask/haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)
source.set(10,50)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(img,1.2,3)  

    for x,y,w,h in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]

        area = w*h
        minarea = 10000
        maxarea = 50000

        if area > minarea and area<maxarea:
            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        if labels_dict[label] == 'MASK':
            val = True

        elif labels_dict[label] == 'NO MASK':
            val = False
        
    if val == True:
        Ds = 'Open'
        print('DOOR STATUS :',Ds)
        val = None

    elif val == False:
        Ds = 'Closed'
        print('DOOR STATUS :',Ds)
        val = None

    else:
        print('DOOR STATUS : No person detected')


    cv2.imshow('MASKER',img)
    key=cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
        
cv2.destroyAllWindows()
source.release()