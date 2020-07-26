import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist
from person_detector import detect_person
import imutils

labels = open('yolo-coco/coco.names').read().strip().split('\n')

net = cv.dnn.readNetFromDarknet('yolo-coco/yolov3.cfg','yolo-coco/yolov3.weights')

ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

for i in range(1,5):
    path = 'pics/pic'+str(i)+'.png'
    frame = cv.imread(path)
    frame = imutils.resize(frame, 700)
    result = detect_person(frame, net, ln, labels.index('person'))

    violaters = set()

    if len(result) >= 2:
        centroids = np.array([r[2] for r in result])
        D = dist.cdist(centroids, centroids, metric='euclidean')

        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):

                if D[i][j] <= 50:
                    violaters.add(i)
                    violaters.add(j)

    for (i, (confi, box, centroid)) in enumerate(result):
        (endX, endY, startX, startY) = box
        (cx, cy) = centroid
        color = (0, 255, 0)

        if i in violaters:
            color = (0, 0, 255)

        cv.rectangle(frame, (endX, endY), (startX, startY), color, 1)
        cv.circle(frame, (cx, cy), 5, color, 2)

    text = "Number of Violaters in the Frame:{}".format(len(violaters))
    cv.putText(frame, text, (10, frame.shape[0] - 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv.imshow("Window",frame)
    cv.waitKey(0)
    cv.imwrite('pics/result'+str(i)+'.PNG',frame)

cv.destroyAllWindows()



