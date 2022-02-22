from itertools import product
import matplotlib.pyplot as plt
import numpy as np

from expe import main

####################################
# run the expe for various (eps0, n)
####################################

ns = [5, 10, 20, 50, 100, 200, 400]
eps0s = [0.1, 0.4]

for (eps0,n) in product(eps0s,ns):
	print("current (n, eps0): ", n, eps0)
	main(int(n),eps0, True, prefix="")

prefix = ""

lss=['-', '--']

def plot_stats(ns, stats, label, color, ls):
    stats = np.array(stats)

    y = stats[:, 1]/stats[:, 2]
    mini = stats[:, 0]
    maxi = stats[:, 3]

    y_err = np.array([y - mini, maxi-y])

    plt.errorbar(ns, y, yerr=y_err, label=label,color=color, ls = ls, capsize=5)

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(8, 10))


for j, current_eps in enumerate(eps0s):
	stats_smart = np.zeros((len(ns), 4))
	stats_local = np.zeros((len(ns), 4))


	stats_smart[:, 0] = float("+inf")
	stats_smart[:, 3] = float("-inf")

	stats_local[:, 0] = float("+inf")
	stats_local[:, 3] = float("-inf")

	for i, n in enumerate(ns):
		n = int(n)

		stats_all_runs = np.load('result/'+prefix+'final'+str(n)+'-'+str(current_eps)+'-'+'T.npy', allow_pickle=True)
		print(stats_all_runs.shape)
		stats_smart[i, 0] = min(stats_smart[i, 0], np.min(stats_all_runs[:, :, :, 0]))
		stats_smart[i, 1] += np.sum(stats_all_runs[:, :, :, 1])
		stats_smart[i, 2] += np.sum(stats_all_runs[:, :, :, 2])
		stats_smart[i, 3] = max(stats_smart[i, 3], np.max(stats_all_runs[:, :, :, 3]))

		stats_all_runs = np.load('result/'+prefix+'localeps'+str(n)+'-'+str(current_eps)+'-'+'T.npy', allow_pickle=True)
		stats_local[i, 0] = min(stats_local[i, 0], np.min(stats_all_runs[:, :, 0]))
		stats_local[i, 1] += np.sum(stats_all_runs[:, :, 1])
		stats_local[i, 2] += np.sum(stats_all_runs[:, :, 2])
		stats_local[i, 3] = max(stats_local[i, 3], np.max(stats_all_runs[:, :, 3]))

	plot_stats(ns, stats_smart, label=r"Network DP ($\epsilon_0 = $"+str(current_eps)+")", color="g", ls=lss[j])
	plot_stats(ns, stats_local, label=r"Local DP ($\epsilon_0 = $"+str(current_eps)+")", color="b", ls=lss[j])



plt.legend(loc='upper right')
plt.xlabel("n")
plt.ylabel("privacy loss")
plt.yscale('log')
ax.tick_params(which="both", axis="y", direction="in")

plt.savefig("fig1b.pdf",bbox_inches='tight', pad_inches=0)
plt.show()

