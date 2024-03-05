import cv2
import numpy as np 
import time
from retinaface import RetinaFace


capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)
capture_time = time.time() + 5
now = time.time()+3
while(True):
	ret, frame = capture.read() 
	frame = cv2.flip(frame, 1) # flip camera verticalyy 
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.imshow('frame', frame)
	# cv2.imshow('gray', gray)

	k = cv2.waitKey(30) & 0xff
	if time.time() >= now:
		cv2.imwrite((f"/home/mohe/Desktop/test/{time.time()}.jpg"), frame)
		now += 3
		

	print(frame)
	if k == 27: #press 'ESC' to exit 
		break

capture.release()
cv2.destroyAllWindows()


