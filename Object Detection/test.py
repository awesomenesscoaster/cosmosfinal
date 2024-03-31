import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from gtts import gTTS
from playsound import playsound
from distance import Distance_finder
import winsound

video = cv2.VideoCapture(0)
labels = []

while True:
    ret, frame = video.read()
    bbox, label, conf = cv.detect_common_objects(frame, model='yolov4-tiny')
    output_image = draw_bbox(frame, bbox, label, conf)

    cv2.imshow("Object Detection", output_image)
    
    for item in label:
        if item in labels:
            pass    
        else:
            labels.append(item)
    
    # for i in range(0,len(bbox)):
    #     x = slice(i,i+1)
    #     bbox_split = bbox[x]
    #     bbox_index = bbox_split[0]    
                                   
    #     bbox_x = bbox_index[2]
    #     bbox_y = bbox_index[3]
    #     dist = Distance_finder(1171.58, float(bbox_y), float(bbox_x))
    #     print(dist)
        
    #     # Link if it gets deleted: "C:\Users\Rhyan Shah\Downloads\timer_beep.mp3"
        
    #     if 14 <= dist < 24:
    #         playsound('C:/Users/Rhyan Shah/Downloads/timer_beep.mp3')
    #         break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
