# -*- coding: utf-8 -*-
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
imObjs = sorted(imObjs, key = lambda o: o[0][0])

# imgWithContours = img.copy()
# drawContourSetsOnImage(contourSets, imgWithContours)
# displayImageGrid([imgbin, imgWithContours], 1, plotImage)
#
# plt.figure()
# displayImageGrid(([imo[1] for imo in imObjs]), 1, plotImage)

alphabet = ['a','s','d','f','g','h','j','k','l','č',
'ć','ž','š','p','o','i','u','z','t','r','e','c','v','b','n','m','đ']

ann = nn.createAnn(len(alphabet))

trainingInput = [nn.transformImageForAnn(obj[1]) for obj in imObjs]
nn.trainAnn(ann, trainingInput, nn.getStandardOutputVectors(len(trainingInput)))

#---------------------------------------------------------------------------

img = loadImage('img/test2.jpg')

imgbin = getBinaryImage(getGrayscaleImage(img))
imgbin = 255 - dilate(erode(imgbin,1),1)

contours = getImageContours(imgbin)
contourSets = geom.mergeContours(contours, mergCrit)
imObjs = getBinaryImageObjectsFromContourSets(imgbin, contourSets)

#------------------------------
angle, center = geom.getBinaryImgObjectsAngleAndCenter(imObjs)
imgbin = getRotatedImage(imgbin, angle, center)
imgbin = getBinaryImage(imgbin)
#------------------------------



contours = getImageContours(imgbin)
contourSets = geom.mergeContours(contours, mergCrit)
imObjs = getBinaryImageObjectsFromContourSets(imgbin, contourSets)

imObjs = sorted(imObjs, key = lambda o: o[0][0])

displayImageGrid(([imo[1] for imo in imObjs]), 1, plotImage)
plt.show()

rowImObjs = imObjs
distances = np.array(geom.getConsecutiveXDistancesBetweenRects(geom.imObjToRects(rowImObjs)))
kmeans = geom.getKMeans(2, distances.reshape(len(distances), 1))
classificationInput = [nn.transformImageForAnn(obj[1]) for obj in rowImObjs]
outputIndices = nn.classify(ann, classificationInput)
print nn.getStringOutputWithSpaces(outputIndices, kmeans.labels_, alphabet)
