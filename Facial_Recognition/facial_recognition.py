import cv2
import os
import np

scanning_process = False
guessing_process = False

names = []

user_name = input("What is your name? ")

if user_name not in names:
    scanning_process = True
    names.append(user_name)
    
    folder_path = os.path.join('C:/Users/Rhyan Shah/Documents/GitHub/cosmosfinal/Facial_Recognition/Datasets', user_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
else:
    guessing_process = True
    print('Welcome back:', user_name)

if scanning_process == True:
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Facial Recognition")
    
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        
        if not ret:
            print("Failed.")
            
        cv2.imshow("Facial Recognition", frame)
        
        k = cv2.waitKey(1)
        
        if k%256 == 27:
            print("Escape")
            break   
        elif k%256 == 32:
            fo = os.chdir(folder_path)
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name,frame)
            print("screenshot taken")            
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()
    
    dir_files = []
    dir_files_cropped = []
    
    dir_index = 'C:/Users/Rhyan Shah/Documents/GitHub/cosmosfinal/Facial_Recognition/Datasets'
    
    for dir in os.scandir(dir_index):
        if dir.is_dir():
            dir_files.append(dir)
            
    for i in range(0,len(dir_files)):
        str(dir_files[i]).replace('<DirEntry', '', inplace = True)
        str(dir_files[i]).replace('>', '', inplace = True)
    
    print(dir_files)
        


# if scanning_process == True:
#     cap = cv2.VideoCapture(0)
#     window_name = 'frame'
#     delay = 1
#     cycle = 300
    
    
#     def save_frame_camera_cycle(device_num, dir_path, basename, cycle, ext='jpg', delay=1, window_name='frame'):
#         if not cap.isOpened():
#             return

#         os.makedirs(dir_path, exist_ok=True)
#         base_path = os.path.join(dir_path, basename)

#         n = 0
    
#     while True:
#         ret, frame = cap.read()
#         cv2.imshow(window_name, frame)
#         if cv2.waitKey(delay) & 0xFF == ord('q'):
#             break
#         if n == cycle:
#             n = 0
#             cv2.imwrite('{}_{}.{}'.format(base_path, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'), ext), frame)
#         n += 1

#         cv2.destroyWindow(window_name)
        
#     save_frame_camera_cycle(0, 'data/temp', 'camera_capture_cycle', 300)
    
#     cap.release()
#     cv2.destroyAllWindows()