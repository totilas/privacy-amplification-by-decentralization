import numpy as np
import matplotlib.pyplot as plt
from composition import advancedCompo
####################################################################
# theoretical bounds expressed for the random walk in Gaussian case
####################################################################


eps0s = [0.1]

ns = np.logspace(0, 10, num=30)
ms=3

delta = 1e-7


def optiCompo(eps, delta, k):
	# return the optimal composition between the naive and advanced formula
	if k < 1e5:
		a = advancedCompo(eps, delta, k)
		b = eps * k
		return min(a, b)
	else:
		return advancedCompo(eps, delta, k)

def Nvertex(N, T, delta, fixed=False):
	# compute the upper bound on the number of vertices
	if fixed:
		return 100
	return T/N+ np.sqrt(3*T/2 * np.log(1/delta))

def localN(N, T, delta, eps,fixed=False ):
	# compute the LDP privacy loss
	Nv = Nvertex(N, T, delta, fixed=fixed)
	return optiCompo(eps, delta, Nv)



def vertexN(N,T, delta, eps, fixed=False):
	# compute the NDP privacy loss
	# c is manually tune (not efficient but easy to trust)
	Nv = Nvertex(N, T, delta, fixed=fixed)
	eps_c = 3*eps/np.sqrt(N)
	return optiCompo(eps_c, delta, Nv+T/N)



if __name__ == "__main__":
	

	save = False
	plt.rcParams.update({'font.size': 22})
	plt.figure(figsize=(8,10))
	
	theo_local = np.zeros((len(ns), len(eps0s)))
	theo_vertex = np.zeros((len(ns), len(eps0s)))

	theo_local_fixed = np.zeros((len(ns), len(eps0s)))
	theo_vertex_fixed = np.zeros((len(ns), len(eps0s)))
	
	for i,n in enumerate(ns):
		print("current n: ", n)
		for j,eps0 in enumerate(eps0s):
			theo_local[i][j] = localN(n, n*100, delta, eps0, fixed=False)
			theo_vertex[i][j] = vertexN(n, n*100, delta, eps0, fixed=False)
			if theo_local[i][j] < theo_vertex[i][j]:
				print("not yet")
			theo_local_fixed[i][j] = localN(n, n*100, delta, eps0, fixed=True)
			theo_vertex_fixed[i][j] = vertexN(n, n*100, delta, eps0, fixed=True)
			if theo_local_fixed[i][j] < theo_vertex_fixed[i][j]:
				print("not yet for fixed")

	if save:

		np.save('result/theo_local', theo_local)
		np.save('result/theo_vertex', theo_vertex)


	else:
		b = ['b','xkcd:sky blue']
		g = ['g', 'xkcd:sea green']
		for i,e in enumerate(eps0s):
			plt.plot(ns, theo_local[:,i], label=r"LDP ($\epsilon_0 = $"+str(e)+")", marker="+", color=b[i], markersize=3)
			plt.plot(ns, theo_vertex[:,i], label=r"NDP ($\epsilon_0 = $"+str(e)+")", marker="+", color=g[i], markersize=3)


			plt.plot(ns, theo_local_fixed[:,i], label=r"LDP ($\epsilon_0 = $"+str(e) + ",fixed contribution)", marker="+", color=b[i], markersize=3, ls='--')
			plt.plot(ns, theo_vertex_fixed[:,i], label=r"NDP ($\epsilon_0 = $"+str(e)+",fixed contribution)", marker="+", color=g[i], markersize=3, ls='--')
		plt.xlabel("n")
		plt.ylabel("privacy loss")
		plt.xscale('log')
		plt.yscale('log')
		plt.legend()
		plt.savefig('fig1a.pdf',bbox_inches='tight', pad_inches=0)
		plt.show()

