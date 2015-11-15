from imageUtils import *
import neuralNet as nn
import numpy as np 
import geomUtil as geom

img = loadImage('img/rotirano2.png')
imgbin = getBinaryImage(getGrayscaleImage(img))
imgbin = dilate(erode(imgbin,1),1)

imObjs = getBinaryImageObjects(imgbin)

displayImageGrid([imo[1] for imo in imObjs], 5, plotImage)

angle, center = geom.getBinaryImgObjectsAngleAndCenter(imObjs)
dst = getRotatedImage(imgbin, angle, center)
dst = getBinaryImage(dst)

plt.figure()

plotImage(dst)

plt.figure()
plotImage(img)
plt.show()


