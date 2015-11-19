class UnionFind(object):
	def __init__(self, length):
		self.labelarray = range(0,length)

	def union(self, i,j):
		iLabel = self.labelarray[i]
		jLabel = self.labelarray[j]

		for i in range(0, len(self.labelarray)):
			if self.labelarray[i] == jLabel:
				self.labelarray[i] = iLabel

	def find(self, i,j):
		return self.labelarray[i] == self.labelarray[j]

	def getLabelArray(self):
		return self.labelarray

