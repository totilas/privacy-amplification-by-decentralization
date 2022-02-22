import numpy as np

def update(a, b):
	if a == None:
		return  [b]
	else :
		a.append(b)
		return a

class Walk():
	def __init__(self, n=1000, K=3000):
		self.n = n
		self.K = K

	def get_K(self):
		return self.K

	def get_n(self):
		return self.n

	def walk_one_step(self, current_vertex):
		raise NotImplementedError

	def relation_matrix(self):
		raise NotImplementedError


class Complete(Walk):


	def __init__(self,n,K):
		super(Complete, self).__init__(n, K)

		self.last_time_passage = np.zeros(n, dtype=int)
		self.last_send = np.zeros(n, dtype=int)
		self.commun_cycle = np.empty((n,n),dtype=object)

	def walk_one_step(self, current_vertex):
		return np.random.randint(self.n)

	def relation_matrix(self):
		current_vertex = 0
		for i in range(1,self.K):
			new_vertex = self.walk_one_step(current_vertex)
			cycle_size = i - self.last_time_passage[new_vertex]


			# add the potential privacy loss for all other vertices
			for v in range(self.n):
				self.commun_cycle[new_vertex][v] = update(self.commun_cycle[new_vertex][v], [cycle_size, 0])	
			# correct the situation for the first and last vertex seen
			self.commun_cycle[new_vertex][self.last_send[new_vertex]][-1][1] += 1
			self.commun_cycle[new_vertex][current_vertex][-1][1] += 1

			# update other parameters
			self.last_send[current_vertex] = new_vertex
			self.last_time_passage[new_vertex] = i
			current_vertex = new_vertex

		return self.commun_cycle




