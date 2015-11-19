from sklearn.cluster import KMeans
import cv2
from scipy import stats
import numpy as np 
from unionFind import *
from misc import *

def getBinaryImgObjectsAngleAndCenter(imObjs):
	xs = [io[0][0] for io in imObjs]
	ys = [io[0][1] for io in imObjs]

	avgx = sum(xs)/len(xs)
	avgy = sum(ys)/len(ys)

	slope, intercept = stats.linregress(xs,ys)[:2]
	angle = -(180/np.pi)*np.arctan(slope)

	return angle, (avgx, avgy) 

def intervalDistance(al, ar, bl, br):
    if al < bl:
        return bl - ar
    else:
        return al - br

def getContourSetRect(cs):

	x = [cv2.boundingRect(c)[0] for c in cs]
	y = [cv2.boundingRect(c)[1] for c in cs]
	w = [cv2.boundingRect(c)[2] for c in cs]
	h = [cv2.boundingRect(c)[3] for c in cs]

	right = max([ z[0] + z[1] for z in zip(x,w)])
	bottom = max([ z[0] + z[1] for z in zip(y,h)])
	x = min(x)
	y = min(y)

	return x,y , right - x, bottom - y

def rectDistance(a,b):
    ax,ay,aw,ah = a
    bx,by,bw,bh = b

    xdist = intervalDistance(ax, ax + aw, bx, bx + bw)
    ydist = intervalDistance(ay, ay + aw, by, by + bw)

    return xdist, ydist

def getConsecutiveXDistancesBetweenRects(rects):
	"""rects are x,y,w,h tuples"""

	rectsShiftedLeft = rects[1:]
	rectsWithoutLast = rects[:-1]
	
	return [t[0][0] - (t[1][0] + t[1][2]) for t in zip(rectsShiftedLeft, rectsWithoutLast)] 

def imObjToRects(imObjs):
	return [(i[0][0], i[0][1], i[1].shape[1], i[1].shape[0]) for i in imObjs]

def mergeContours(contours, criterion):

	contourPartition = UnionFind(len(contours))

	for i in xrange(0,len(contours)):
		for j in range(i+1, len(contours)):
			if not contourPartition.find(i,j) and criterion(contours[i], contours[j]):
				contourPartition.union(i,j)

	return partition(contours, contourPartition.getLabelArray())

def getKMeans(k, data):
	"""objekat ima labels_ i cluster_centers_"""

	kMeans = KMeans(n_clusters=k, max_iter=2000, tol=0.00001, n_init=10)
	kMeans.fit(data)
	return kMeans
