from imageUtils import *
import neuralNet as nn
import numpy as np 
import geomUtil as geom

def mergCrit(a, b):
    ar = cv2.boundingRect(a)
    br = cv2.boundingRect(b)
    xdist = geom.rectDistance(ar, br)[0]
    maxw = max(ar[2], br[2])
    
    return xdist < 0 and maxw*0.5 < abs(xdist)

#######################################################

img = loadImage('img/obucavajuciSkup.jpg')

imgbin = getBinaryImage(getGrayscaleImage(img))
imgbin = dilate(erode(imgbin,1),1)

modifyPixels(imgbin, lambda p: 255 - p)
contours = getImageContours(imgbin)
contourSets = geom.mergeContours(contours, mergCrit)

imObjs = getBinaryImageObjectsFromContourSets(imgbin, contourSets)

imgWithContours = img.copy()
drawContourSetsOnImage(contourSets, imgWithContours)
displayImageGrid([imgbin, imgWithContours], 1, plotImage)

plt.figure()
displayImageGrid(([imo[1] for imo in imObjs]), 1, plotImage)

plt.show()
