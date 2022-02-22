import numpy as np



def naiveCompo(list_eps):
	return sum(list_eps)


def advancedCompo(eps, delta, k):
	return np.sqrt(2*k*np.log(1/delta))*eps + k * eps * (np.exp(eps)-1)


def kairouzCompo(list_eps, delta):
	if len(list_eps) == 0:
		new_delta = delta
	new_delta = len(list_eps)*delta # arbitrary choice to have equal contribution to the delta
	if list_eps == None:
		return 0
	else:
		list_eps = np.array(list_eps)
		worst = np.sum(list_eps)
		first_term = np.sum((np.exp(list_eps) - 1) * list_eps / (np.exp(list_eps)+1))
		squared_eps = np.sum(list_eps*list_eps)
		if np.exp(1) + np.sqrt(squared_eps) / new_delta > 1/new_delta:
			second_term = np.sqrt(2* squared_eps * np.log(1/new_delta))

		else :
			second_term = np.sqrt(2* squared_eps * np.log(np.exp(1)+ np.sqrt(squared_eps)/new_delta))
		total = first_term + second_term
		if total > worst :
			return worst
		else:
			return total


