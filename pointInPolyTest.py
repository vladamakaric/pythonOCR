import cv2
import numpy as np 

contour = [ (0,0), (100,100), (100,0)] 

x,y,w,h = cv2.boundingRect(np.array(contour))

print cv2.pointPolygonTest(np.array(contour), (15,10), False)

contour2 = [ (220,200), (400,200), (400,400), (200,400) ] 

c = np.concatenate((np.array(contour), np.array(contour2)))

print cv2.pointPolygonTest(c, (300,600), False)
