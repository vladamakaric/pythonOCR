from sklearn.cluster import KMeans
from scipy import stats
import numpy as np 

def getBinaryImgObjectsAngleAndCenter(imObjs):
	xs = [io[0][0] for io in imObjs]
	ys = [io[0][1] for io in imObjs]

	avgx = sum(xs)/len(xs)
	avgy = sum(ys)/len(ys)

	slope, intercept = stats.linregress(xs,ys)[:2]
	angle = -(180/np.pi)*np.arctan(slope)

	return angle, (avgx, avgy) 


def partition(arr, partitionIndex, numArrs):
	partitionDic = {}

	for el in zip(arr, partitionIndex):
		if el[1] not in partitionDic:
			partitionDic[el[1]] = []

		partitionDic[el[1]].append(el[0]) 

	dicItems = sorted(partitionDic.items(), key= lambda i: i[0])

	values = zip(*dicItems)[1]

	return values

def getConsecutiveXDistancesBetweenRects(rects):
	"""rects are x,y,w,h tuples"""

	rectsShiftedLeft = rects[1:]
	rectsWithoutLast = rects[:-1]
	
	return [t[0][0] - (t[1][0] + t[1][2]) for t in zip(rectsShiftedLeft, rectsWithoutLast)] 

def imObjToRects(imObjs):
	return [(i[0][0], i[0][1], i[1].shape[1], i[1].shape[0]) for i in imObjs]

def getKMeans(k, data):
	"""objekat ima labels_ i cluster_centers_"""

	kMeans = KMeans(n_clusters=k, max_iter=2000, tol=0.00001, n_init=10)
	kMeans.fit(data)
	return kMeans
