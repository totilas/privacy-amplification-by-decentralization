import findSigma
import data
import private

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import typer


app = typer.Typer()

###############################################################################
# Set the parameters
###############################################################################


def main(
    n_nodes: int=2000,
    eps_tot: float=10,
    delta: float=1e-6,
    n_iter: int=20000,
    conf: float=1.25,
    L: float=0.4,
    seed: int=1,
    n_trials: int=1,
    optimize_gamma: bool=False,
    save_array: bool=False,
    save_fig: bool=False,
    plot_fig: bool=True
    ):

    assert 0 <= delta <= 1
    assert 0 <= L <= 1

    np.random.seed(seed)

    X_train, X_test, y_train, y_test = data.load("Houses")
    print("Successfully load dataset")



    ###########################################################################
    # Find max sigma
    ###########################################################################

    print("Computing the sigmas:")
    # for local DP, with advanced composition
    sigma_loc = findSigma.loc(L, n_nodes, eps_tot, delta, n_iter)

    # for central DP, basic DPSGD result based on Bassily et al.
    sigma_ref = findSigma.dpsgd(L, n_nodes, eps_tot, delta, n_iter)

    # for network DP, bound with paper result
    sigma_net = findSigma.net(L, n_nodes, eps_tot, delta, n_iter)




    def score(y):
        # defining score to be able to evaluate the model on the test set during the training
        def evaluation(theta):

            from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
            class Truc(BaseEstimator, LinearClassifierMixin):
                def __init__(self):
                    self.intercept_ = np.expand_dims(theta[-1], axis=0)
                    self.coef_ = np.expand_dims(theta[:-1], axis=0)
                    self.classes_ = np.unique(y)

                def fit(self, X, y):
                    pass

            truc = Truc()

            return truc.score(X_test, y_test) 

        return evaluation


    ###########################################################################
    # find optimal gamma for the three cases
    ###########################################################################


    n_train = X_train.shape[0]

    best_gamma = np.array([0.06733333, 0.03416667, 0.03416667])

    if optimize_gamma:
        gamma_range = np.linspace(1e-3, .2, num=7)

        print("Testing various gamma", gamma_range)

        for i,sigma in enumerate([sigma_loc, sigma_ref, sigma_net]):
            print("optimizing ", sigma)

            best_objfun = 0
            for gamma in gamma_range:

                n_runs = 6
                objfun = np.zeros(n_runs)
                if sigma != sigma_ref:
                    mlr = private.MyPrivateRWSGDLogisticRegression(gamma, n_iter, n_nodes, sigma_ref, 0, stopping_criteria = "contribute_then_noise",max_updates_per_node = conf *n_iter + n_iter/n_nodes,random_state=None, score=score, freq_obj_eval=1000, L=L)
                else :
                    mlr = private.MyPrivateRWSGDLogisticRegression(gamma, n_iter, n_nodes, sigma_ref, 0, stopping_criteria = "contribute_then_noise",max_updates_per_node = n_iter,random_state=None, score=score, freq_obj_eval=1000, L=L)

                for r in range(n_runs):
                    mlr.fit(X_train, y_train)
                    objfun[r] = mlr.obj_list_[-1]
                if objfun.mean() < best_objfun:
                    best_objfun = objfun.mean()
                    best_gamma[i] = gamma


        print("Found the following :", best_gamma)
        if save_array:
            np.save('result/gamma.txt', best_gamma)


    ###########################################################################
    # Core experiments n_trials runs for the three methods
    ###########################################################################

    freq_eval = 100
    obj_list_ref = np.zeros((n_trials, int(n_iter/freq_eval)))
    obj_list_loc = np.zeros((n_trials, int(n_iter/freq_eval)))
    obj_list_net = np.zeros((n_trials, int(n_iter/freq_eval)))

    score_ref = np.zeros((n_trials, int(n_iter/freq_eval)))
    score_loc = np.zeros((n_trials, int(n_iter/freq_eval)))
    score_net = np.zeros((n_trials, int(n_iter/freq_eval)))



    for i in range(n_trials):
        print("Computing trial ", i)
        # put option contribution the noise, but with a max number of iteration equal to the whole experiment, so we always compute the gradient
        mlr_ref = private.MyPrivateRWSGDLogisticRegression(best_gamma[1], n_iter, n_nodes, sigma_ref, 0, stopping_criteria = "contribute_then_noise",max_updates_per_node = n_iter,random_state=np.random.randint(1000), score=score, freq_obj_eval=freq_eval, L=L)
        mlr_ref.fit(X_train, y_train)

        mlr_loc = private.MyPrivateRWSGDLogisticRegression(best_gamma[0], n_iter, n_nodes, sigma_loc, 0, stopping_criteria = "contribute_then_nothing",max_updates_per_node = conf * n_iter/n_nodes,random_state=np.random.randint(1000), score=score, freq_obj_eval=freq_eval, L=L)
        mlr_loc.fit(X_train, y_train)

        mlr_net = private.MyPrivateRWSGDLogisticRegression(best_gamma[2], n_iter, n_nodes, sigma_net, 0, stopping_criteria = "contribute_then_noise",max_updates_per_node = conf * n_iter/n_nodes,random_state=np.random.randint(1000), score=score, freq_obj_eval=freq_eval, L=L)
        mlr_net.fit(X_train, y_train)


        obj_list_ref[i] = mlr_ref.obj_list_
        obj_list_loc[i] = mlr_loc.obj_list_
        obj_list_net[i] = mlr_net.obj_list_

        score_ref[i] = mlr_ref.scores_
        score_loc[i] = mlr_loc.scores_
        score_net[i] = mlr_net.scores_

    ###########################################################################
    # save objective function and score over iterations
    ###########################################################################

    if save_array:
        np.save("result/dpsgd", obj_list_ref)
        np.save("result/localsgd", obj_list_loc)
        np.save("result/networksgd", obj_list_net)

        np.save("result/dpsgd_score",score_ref)
        np.save("result/localsgd_score",score_loc)
        np.save("result/networksgd_score", score_net)


    ###########################################################################
    # plot figure and save them
    ###########################################################################

    if plot_fig:
        # define x axis
        iter_list = np.arange(len(obj_list_ref[0])) * mlr_ref.freq_obj_eval
        # plot
        plt.plot(iter_list, obj_list_ref.mean(axis=0), label="Centralized DP-SGD", color="r")
        plt.plot(iter_list, obj_list_loc.mean(axis=0), label="Local DP-SGD", color="b")
        plt.plot(iter_list, obj_list_net.mean(axis=0), label="Network DP-SGD", color="g")

        plt.xlabel("Iteration")
        plt.ylabel("Objective function")
        plt.yscale("log")
        plt.legend(loc='upper right')
        if save_fig:
            plt.savefig("result/objfun.pdf")
        plt.show()


        plt.figure()
        plt.plot(iter_list, score_ref.mean(axis=0), label="Centralized DP-SGD", color="r")
        plt.plot(iter_list, score_loc.mean(axis=0), label="Local DP-SGD", color="b")
        plt.plot(iter_list, score_net.mean(axis=0), label="Network DP-SGD", color="g")

        plt.xlabel("Iteration")
        plt.ylabel("Test Accuracy")
        plt.legend(loc='lower right')
        if save_fig:
            plt.savefig("result/accuracy.pdf",bbox_inches='tight', pad_inches=0)

        plt.show()


if __name__ == "__main__":
    typer.run(main)