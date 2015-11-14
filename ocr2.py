from imageUtils import *
import neuralNet as nn
import numpy as np 
import geomUtil as geom

img = loadImage('img/alphabet.png')
imgbin = getBinaryImage(getGrayscaleImage(img))
imgbin = dilate(erode(imgbin,1),1)

imObjs = getBinaryImageObjects(imgbin)
imObjs = sorted(imObjs, key = lambda o: o[0][0])

# imgWithContours = img.copy()
# drawContoursOnImage(imgWithContours, imgbin)
# displayImageGrid([imgbin, imgWithContours], 1, plotImage)

alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

ann = nn.createAnn(len(alphabet))

# plt.figure()
# displayImageGrid([imo[1] for imo in imObjs], 1, plotImage)

trainingInput = [nn.transformImageForAnn(obj[1]) for obj in imObjs]

nn.trainAnn(ann, trainingInput, nn.getStandardOutputVectors(len(trainingInput)))

#------------------------------------------------------------------

img = loadImage('img/LoremIpsum.png')
imgbin = getBinaryImage(getGrayscaleImage(img))
imgbin = dilate(erode(imgbin,1),1)

imgWithContours = img.copy()
drawContoursOnImage(imgWithContours, imgbin)
displayImageGrid([imgbin, imgWithContours], 1, plotImage)

imObjs = getBinaryImageObjects(imgbin)
imObjs = sorted(imObjs, key = lambda o: o[0][0])

distances = np.array(geom.getConsecutiveXDistancesBetweenRects(geom.imObjToRects(imObjs)))
print distances

kmeans = geom.getKMeans(2, distances.reshape(len(distances), 1))

plt.figure()

displayImageGrid([imo[1] for imo in imObjs], 5, plotImage)

classificationInput = [nn.transformImageForAnn(obj[1]) for obj in imObjs]

outputIndices = nn.classify(ann, classificationInput)

print nn.getStringOutputWithSpaces(outputIndices, kmeans.labels_, alphabet)

plt.show()
