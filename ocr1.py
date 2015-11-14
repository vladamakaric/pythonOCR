from imageUtils import *
import neuralNet as nn
import numpy as np 

img = loadImage('alphabet.png')
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

img = loadImage('LoremIpsum.png')
imgbin = getBinaryImage(getGrayscaleImage(img))
imgbin = dilate(erode(imgbin,1),1)

imgWithContours = img.copy()
drawContoursOnImage(imgWithContours, imgbin)
displayImageGrid([imgbin, imgWithContours], 1, plotImage)

imObjs = getBinaryImageObjects(imgbin)
imObjs = sorted(imObjs, key = lambda o: o[0][0])

plt.figure()

displayImageGrid([imo[1] for imo in imObjs], 5, plotImage)

classificationInput = [nn.transformImageForAnn(obj[1]) for obj in imObjs]

outputIndices = nn.classify(ann, classificationInput)

print [alphabet[i] for i in outputIndices]

plt.show()
