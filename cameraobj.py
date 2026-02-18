import cv2

camera = cv2.VideoCapture(0)

while(camera.isOpened):
    ret, frame = camera.read()

    cv2.imshow("camera", frame)

    if cv2.waitKey(1) == ord("q"): # "q" to close the camera object
        break