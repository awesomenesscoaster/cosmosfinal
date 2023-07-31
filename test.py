import cv2
import cvlib as cv

from cvlib.object_detection import draw_bbox
from gtts import gTTS
from playsound import playsound
from distance import Distance_finder

video = cv2.VideoCapture(0)
labels = []

while True:
    ret, frame = video.read()
    bbox, label, conf = cv.detect_common_objects(frame)
    output_image = draw_bbox(frame, bbox, label, conf)

    cv2.imshow("Object Detection", output_image)
    
    for item in label:
        if item in labels:
            pass
        else:
            labels.append(item)
    
    for i in range(0,len(bbox)):
        x = slice(i,i+1)
        bbox_split = bbox[x]
        bbox_index = bbox_split[0]
        print(bbox_split)
        
        for object in frame:
            bbox_x = bbox_index[2]
            bbox_y = bbox_index[3]
            print(Distance_finder(1171.58, float(bbox_x), float(bbox_y)), 'inches.')

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cv2.destroyAllWindows()