from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import time, imutils, dlib, cv2, os
import numpy as np
from keras.models import load_model

val = None
Ds = None
model = load_model('mask/model8.h5')

face_clsfr=cv2.CascadeClassifier('mask/haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)
source.set(10,200)
source.set(3,300)
source.set(4,300)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

net = cv2.dnn.readNetFromCaffe('mobilenet_ssd\MobileNetSSD_deploy.prototxt','mobilenet_ssd\MobileNetSSD_deploy.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

vs = cv2.VideoCapture(r'videos\example_01.mp4') #for stream people counter

W = None
H = None
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
totalFrames = 0
totalDown = 0
totalUp = 0
Ptotal = 0
limit = 10 # set limit

fps = FPS().start()

print('[INFO] Loading..')

while True:
    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(img,1.5,3)
    _, frame = vs.read()
    frame = imutils.resize(frame,width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    (H, W) = frame.shape[:2]


    status = "Waiting"
    rects = []

    if totalFrames % 30 == 0:
        status = "Detecting"
        trackers = []

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.4:
                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                trackers.append(tracker)

    else:
        for tracker in trackers:
            status = "Tracking"

            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))
    
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
    cv2.putText(frame, "-Entrance Border-", (10, H - ((i * 20) + 200)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)

        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

        trackableObjects[objectID] = to

        Ptotal = int(int(totalDown)-int(totalUp))

        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

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


    info = [
    ("Exit", totalUp),
    ("Enter", totalDown),
    ("Status", status),
    ]

    info2 = [
    ("Total people inside", Ptotal),
    ]

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for (i, (k, v)) in enumerate(info2):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    if val == True and Ptotal <= limit:
        Ds = 'Open'
        print('DOOR STATUS :',Ds)
        val = None

    elif val == True and Ptotal > limit:
        Ds = 'Closed'
        print('DOOR STATUS :',Ds)
        val = None

    elif val == False and Ptotal > limit:
        Ds = 'Closed'
        print('DOOR STATUS :',Ds)
        val = None
        
    elif val == False and Ptotal <= limit:
        Ds = 'Closed'
        print('DOOR STATUS :',Ds)
        val = None

    elif val == None and Ptotal > limit:
        Ds = 'Closed'
        print('DOOR STATUS :',Ds)

    else:
        Ds = 'Open'
        print('DOOR STATUS :',Ds)
        print('[NOTE]No person detected, default set Open')

    cv2.imshow('Mask Detection',img)
    cv2.imshow("People Counter", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    totalFrames += 1
    fps.update()


fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] You have turned off this device")

source.release()
vs.release()
cv2.destroyAllWindows()