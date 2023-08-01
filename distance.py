import numpy as np
import imutils
import cv2

from imutils import paths

Known_distance = 36.0
Known_width = 4.25

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

fonts = cv2.FONT_HERSHEY_COMPLEX

# face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
#     focal_length = (width_in_rf_image * measured_distance) / real_width
#     return focal_length

def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = ((real_face_width * Focal_Length)/face_width_in_frame)/96 * 1.714
    return distance

# def face_data(image):
#     face_width = 0
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
#     for (x, y, h, w) in faces:
#         cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
#         face_width = w
#     return face_width

# ref_image = cv2.imread("/Users/Rhyan Shah/Downloads/IMG-5259.jpg")
# ref_image_face_width = face_data(ref_image)

# Focal_length_found = Focal_Length_Finder(
#     Known_distance, Known_width, ref_image_face_width)

# print(Focal_length_found)

# cv2.imshow("/Users/Rhyan Shah/Downloads/IMG-5259.jpg")

# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
    
#     face_width_in_frame = face_data(frame)
    
#     if face_width_in_frame != 0:
#         Distance = Distance_finder(
#             Focal_length_found, Known_width, face_width_in_frame)
        
#         cv2.line(frame, (30,30), (230,30), RED, 32)
#         cv2.line(frame, (30, 30), (230, 30), BLACK, 28)
         
#         cv2.putText(
#             frame, f"Distance: {round(Distance,2)} CM", (30, 35), 
#           fonts, 0.6, GREEN, 2)
        
#     cv2.imshow("frame", frame)
        
#     if cv2.waitKey(1) == ord("q"):
#         break
    
# cap.release()
# cv2.destroyAllWindows()
    
# OPTION 3
# def find_marker(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5,5), 0)
#     edged = cv2.Canny(gray, 35, 125)
    
#     cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     c = max(cnts, key = cv2.contourArea)
    
#     return cv2.minAreaRect(c)

# def distance_to_camera(knownWidth, focalLength, perWidth):
#     return (knownWidth * focalLength) / perWidth

# KNOWN_DISTANCE = 36.0
# KNOWN_WIDTH = 4.25

# image = cv2.imread("/Users/Rhyan Shah/Downloads/IMG-5259.jpg")
# marker = find_marker(image)
# focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
# print(focalLength)

# for imagePath in sorted(paths.list_images("images")):
#     image = cv2.read(imagePath)
#     marker = find_marker(image)
#     inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])