from unittest.mock import patch
from darkflow.net.build import TFNet
from patch1 import *
import numpy as np
import cv2
import csv
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

def boxing(original_img, predictions):
    newImage = np.copy(original_img)
    car = 0
    truck = 0
    motorbike = 0
    for result in predictions:
        if result['label'] == 'car' or result['label'] == 'motorbike' or result['label'] == 'truck':
            if result['label'] == 'car':
                car = car + 1
            if result['label'] == 'motorbike':
                motorbike = motorbike + 1
            if result['label'] == 'truck':
                truck = truck + 1


            top_x = result['topleft']['x']
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            confidence = result['confidence']
            label = result['label'] + " " + str(round(confidence, 3))

            if confidence > 0.3:
                newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 2)

    print ("cars : {}".format(car))
    print ("trucks : {}".format(truck))
    print ("motorbike : {}".format(motorbike))
    print ('\n')
    return newImage



#options = {"model": "cfg/tiny-yolo-4c.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1, "gpu": 1.0}
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.3}

tfnet = TFNet(options)
fps=25
print("Enter the type of input")
opt=str(input())

while True:
    # opt = str(input('#>>>'))
    if opt == 'image':
        imgcv = cv2.imread("sample_img/sample_dog.jpg")
        result = tfnet.return_predict(imgcv)
        print(result)

    elif opt == 'video':
         cap = cv2.VideoCapture('video.mp4')
         width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
         height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

         fourcc = cv2.VideoWriter_fourcc(*'DIVX')
         out = cv2.VideoWriter('output2.mp4',fourcc, 20.0, (int(width), int(height)))
        #  writer = cv2.VideoWriter('video_result.mp4',cv2.VideoWriter_fourcc(*'DIVX'),fps,(int(width)),int(height))
         count = 0
         while True:
            if cap.isOpened():
                ret, frame = cap.read()
                # cv2.imshow('frame',new_frame)
                if ret == True:
                    print ("frame count : {}".format(count))
                    frame = np.asarray(frame)
                    results = tfnet.return_predict(frame)
                    # new_frame = boxing(frame, results)
                    for v in results:
                        new_frame = boxing(frame, results)
                        # Display the resulting frame
                        out.write(new_frame)
                    cv2.imshow('frame',new_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


                else:
                    frame = np.asarray(frame)
                    out.write(frame)

            else:
                break
            count = count + 1

            cap.release()
            out.release()
            cv2.destroyAllWindows()

    elif opt == 'exit':
        print ('exiting....')
        break
