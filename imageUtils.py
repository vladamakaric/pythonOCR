import cv2
import numpy as np 
import matplotlib.pyplot as plt

def dilate(image, i):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=i)

def erode(image, i):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=i)

def invert(image):
    return 255-image

def loadImage(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def getGrayscaleImage(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def plotImage(img, color = False):
    plt.imshow(img) if color else plt.imshow(img, 'gray')

def displayImageGrid(images, cols, plotFunc):
	rows = np.ceil(len(images) / float(cols))

	for i, img in enumerate(images): 
		
		plt.subplot(rows, cols, i+1)
		plotFunc(img)
        
def getRotatedImage(img, angle, center):
	rows, cols = img.shape[:2]
	M = cv2.getRotationMatrix2D(center,-angle,1)
	return cv2.warpAffine(img,M,(cols,rows))

def getBinaryImage(gsImg):
    return cv2.threshold(gsImg, 127, 255, cv2.THRESH_BINARY)[1]


def getBinaryImageObjects(binImg):
    contours, hierarchy = cv2.findContours(binImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    def getObjectFromContour(contour):
        x,y,w,h = cv2.boundingRect(contour)
        region = np.zeros((h,w))


        #piksel koordinate globalne
        def putPixel(i,j):
            ppt = cv2.pointPolygonTest(contour, (i,j), False)
            if ppt>=0 and binImg[j,i] == 255:
                return 255 
            else: 
                return 0

        region = np.array([putPixel(x+j, y+i) for i in xrange(0,h) for j in xrange(0,w)])
        region = np.reshape(region, (h,w))

        return (x, y), region

    return [getObjectFromContour(c) for c in contours]

def drawContoursOnImage(img, binImg):
	contours, hierarchy = cv2.findContours(binImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for c in contours:
		r,g,b = np.random.randint(0,2,3)
		cv2.drawContours(img, [c], -1, (r*255,g*255,b*255), 2)


