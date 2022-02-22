import numpy as np
import matplotlib.pyplot as plt

from statistics import mean
import collections

from amplification import *
from mechanisms import LDPMechanism
from randomWalk import *
from composition import *
from itertools import product

from functools import lru_cache, wraps

def memoise(f):
	d = dict()

	@wraps(f)
	def wrapper(*args):
		if args not in d:
			d[args] = f(*args)
		
		return d[args]

	return wrapper

def stats(l):
	if len(l) > 0:
		return min(l), sum(l), len(l), max(l)
	else:
		return float("+inf"), 0, 0, float("-inf")


def main(n=100, eps0=0.1, gaussian=False, prefix=""):
	####################################################################
	# generate the walk
	####################################################################

	K = 100*n

	nb_runs = 10

	delta = 1e-7

	if gaussian:
		suffix = "T"
	else:
		suffix = "F"


	####################################################################
	# transform the paths into epsilons
	####################################################################

	# define some useful mechanisms
	generic = LDPMechanism()
	erlingsson = Erlingsson()
	hoeffding = Hoeffding(generic)
	bennett = BennettExact(generic)
	subsampling = Subsampling()
	aggregat = Aggregation()

	@memoise
	def translation(size_cycle, contrib, gaussian=False):
		# Function that computes the eps for a given situation

		if gaussian:
			mass_eps = aggregat.get_eps(eps0, size_cycle)
		else:
			mass_eps = min(
				hoeffding.get_eps(eps0, size_cycle, delta),
				bennett.get_eps(eps0, size_cycle, delta),
				erlingsson.get_eps(eps0, size_cycle, delta)
			)

		if contrib == 0:
			# we have no indication of the participation of the vertex
			# we model it as a shuffling of a subsampling in general
			eps = subsampling.get_eps(mass_eps, n, size_cycle)

		elif contrib == 1:
			# we know that the node participates for sure, so we do not use subsampling
			eps = mass_eps

		else:
			# we know that the node participates twice for sure, so we double eps for safety
			eps = 2*mass_eps

		return eps



	# last dimension is min, sum, len, max.
	privacy_loss_array = np.zeros((nb_runs, n, n, 4))
	local_eps_array = np.zeros((nb_runs, n, 4))


	for i in range(nb_runs):



		my_walk = Complete(n, K)
		study = my_walk.relation_matrix()

		# convert the list of cycles into list of eps
		relation_eps = np.empty((n,n), dtype = object)

		for (u,v) in product(range(n), repeat=2):
			list_cycle = study[u][v]
			list_eps = []
			if list_cycle == None:
				# there is no relation between both vertex
				pass
			else:
				for cycle in list_cycle:
					size_cycle, contrib = cycle[0], cycle[1]
					list_eps.append(translation(size_cycle, contrib, gaussian))
			relation_eps[u][v] = list_eps

		####################################################################
		# compute composition of the epsilon
		####################################################################
		# we use the composition on the previous list to define the total privacy loss
		# for each pair of vertices
		privacy_loss = collections.defaultdict(list)
		for (u,v) in product(range(n), repeat=2):
			# suppress the privacy loss due to communication with itself
			if u == v:
				continue

			privacy_loss[u,v].append(kairouzCompo(relation_eps[u][v], delta))



		####################################################################
		# comparison between local DP and vertex DP
		####################################################################
		# we compute the local DP for each vertex
		local_eps = collections.defaultdict(list)
		for v in range(n):
			number_contrib = len(relation_eps[v][0])
			local_eps[v].append(advancedCompo(eps0, delta, number_contrib))

	
		for (u, v) in product(range(n), repeat=2):
			privacy_loss_array[i, u, v] = stats(privacy_loss[u, v])

		for u in range(n):
			local_eps_array[i, u] = stats(local_eps[u])

	# final privacy loss
	np.save('result/'+prefix+'final'+str(n)+'-'+str(eps0)+'-'+suffix, privacy_loss_array)

	# local dp
	np.save('result/'+prefix+'localeps'+str(n)+'-'+str(eps0)+'-'+suffix, local_eps_array)


