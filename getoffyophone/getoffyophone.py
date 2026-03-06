import cv2
import random
import time
import os
import pygame
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.to = ("cuda")

script_dir = os.path.dirname(os.path.abspath(__file__))
imagesPath = [
    os.path.join(script_dir, 'images', '1.jpg'),
    os.path.join(script_dir, 'images', '2.jpg'),
    os.path.join(script_dir, 'images', '3.jpg'),
    os.path.join(script_dir, 'images', '4.jpg'),
    os.path.join(script_dir, 'images', '5.png')
]
images = []
for img_path in imagesPath:
    img = cv2.imread(img_path)
    if img is not None:
        images.append(img)
    else:
        print(f"Warning: Could not load image at {img_path}")

alarm = os.path.join(script_dir, 'audio', 'alarm.mp3')
vine_boom = os.path.join(script_dir, 'audio', 'vine-boom.mp3')

# initialize pygame mixer for audio
pygame.mixer.init()
vine_boom_sound = pygame.mixer.Sound(vine_boom)

camera = cv2.VideoCapture(0) # camera object
opened_windows = []  # track opened image windows

knownObject = dict()
def drawDetections(img, detections, threshold):
    boxes = detections.boxes

    for box in boxes:
        if float(box.conf[0]) > threshold:
            objClass = int(box.cls[0])
            if objClass not in knownObject.keys():
                knownObject[objClass] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), knownObject[objClass], 2)
            
            # draw label with confidence
            label = f"{model.names[objClass]}: {float(box.conf[0]):.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, knownObject[objClass], 2)
    
    return img

while(camera.isOpened()):
    ret, frame = camera.read()

    detections = model(frame)[0]
    frame = drawDetections(frame, detections, 0.5)

    cellphone_detected = False
    
    for box in detections.boxes:

        if int(box.cls[0]) == 67 and float(box.conf[0]) > 0.5:  # 67 = phone
                cellphone_detected = True

    while cellphone_detected:
            if not pygame.mixer.music.get_busy(): # start playing alarm if not already playing
                pygame.mixer.music.load(alarm)
                pygame.mixer.music.play(-1)  # -1 means loop infinitely
            
            window_name = f'STOP USING YO PHONE_{time.time()}'
            opened_windows.append(window_name)  # track the window
            random_image = random.choice(images)
            cv2.imshow(window_name, random_image)
            vine_boom_sound.play()  # play vine boom sound when image opens
            cv2.imshow('Object Detection Camera', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'): # new image every 100ms
                break
            
            ret, frame = camera.read()
            detections = model(frame)[0]
            frame = drawDetections(frame, detections, 0.5)
            
            cellphone_detected = False
            for box in detections.boxes:
                if int(box.cls[0]) == 67 and float(box.conf[0]) > 0.5:
                    cellphone_detected = True
                    break
    
    # Stop alarm and close all image windows when phone is no longer detected
    if not cellphone_detected and pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        for window in opened_windows:
            cv2.destroyWindow(window)
        opened_windows.clear()

    cv2.imshow('Object Detection Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()