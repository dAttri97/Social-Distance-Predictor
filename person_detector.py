import numpy as np
import cv2 as cv

def detect_person(frame,net ,ln ,index = 0):
    (H,W) = frame.shape[:2]

    result = []

    blob = cv.dnn.blobFromImage(frame,1/255.0,(416,416),crop=False,swapRB=True)

    net.setInput(blob)
    outputLayers =net.forward(ln)

    boxes,centroids,confindences=[],[],[]

    for output in outputLayers:
        for detection in output:
            score = detection[5:]
            classId = np.argmax(score)
            confindence = score[classId]

            if classId == index and confindence>=0.6:
                box = detection[0:4] * np.array([W,H,W,H])
                centerX, centerY,width,height =box.astype('int')

                x = int(centerX-(width/2))
                y = int(centerY-(height/2))

                boxes.append([x,y,int(width),int(height)])
                centroids.append((centerX,centerY))
                confindences.append(float(confindence))

    idx = cv.dnn.NMSBoxes(boxes,confindences,0.5,0.5)

    if len(idx)>0:
        for i in idx.flatten():
            (x,y) = (boxes[i][0],boxes[i][1])
            (w,h) = (boxes[i][2],boxes[i][3])

            r =(confindences[i],(x,y,x+w,y+h),centroids[i])
            result.append(r)

    return result