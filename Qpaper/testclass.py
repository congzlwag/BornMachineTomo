from numpy import *

class D:
	def __init__(self,n,m):
		self._n = n
		self._m = m
		self.h = 0
		self.r = m
		self._dat = random.rand(n,m).tolist()
	def show(self):
		print(self._dat)
	def chunk_out(self):
		return self._dat[self.h:self.r]


if __name__ == '__main__':
	ds = D(4,5)
	ds.h = 2
	chunk = ds.chunk_out()
	print(chunk)
	chunk[0]=-1
	ds.show()