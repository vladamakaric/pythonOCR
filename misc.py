def partition(arr, partitionIndex):
	partitionDic = {}

	for el in zip(arr, partitionIndex):
		if el[1] not in partitionDic:
			partitionDic[el[1]] = []

		partitionDic[el[1]].append(el[0]) 

	dicItems = sorted(partitionDic.items(), key= lambda i: i[0])

	values = zip(*dicItems)[1]

	return values
