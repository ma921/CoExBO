import ast
import torch
import pandas as pd
import matplotlib.pyplot as plt
from GPSHAP.explanation_algorithms.GPSHAP import GPSHAP
from GPSHAP.utils.visualisation.deterministic_values import summary_plot
from GPSHAP.utils.visualisation.stochastic_values import local_explanation_plot, global_explanation_plot
from GPSHAP.acquisition_functions.ucb import GPUCBSHAP
from GPSHAP.gp_models.ExactGPRegression import ExactGPRegression
from gpytorch.kernels import ScaleKernel, RBFKernel
from ._utils import TensorManager
from ._gp_regressor import predict


class HumanFeedback(TensorManager):
    def __init__(self, feature_names):
        """
        A class for communicating with a human user.
        
        Args:
        - feature_names: list, a list of feature names
        """
        TensorManager.__init__(self)
        self.feature_names = feature_names
        
    def display_pairwise_samples(self, X_pairwise_next, random=False):
        """
        Display the current pairwise samples
        
        Args:
        - X_pairwise_next: torch.tensor, a pairwise candidate
        - random: bool, whether or not the pairwise candidate is randomly generated.
        """
        X0, X1 = torch.chunk(X_pairwise_next, dim=1, chunks=2)
        if random:
            print("X0 (random): " + str(X0.squeeze().float()))
            print("X1 (random): " + str(X1.squeeze().float()))
        else:
            print("X0 (preference): " + str(X0.squeeze().float()))
            print("X1 (normal UCB): " + str(X1.squeeze().float()))
        
    def get_human_feedback(self, rand=False):
        """
        Get the human feedback from the interface
        
        Args:
        - random: bool, whether or not the pairwise candidate is randomly generated.
        
        Return:
        - y_pairwise_next: torch.tensor, human selection results.
        - y_pairwise_unsure_next: torch.tensor, 1 if unsure, otherwise 0.
        """
        human_feedback = int(input('Type 0 or 1 which you think larger. Type 2 if unsure.'))
        if human_feedback == 1:
            feedback_sure = 1
            feedback_unsure = 1
            print("You chose X1")
        elif human_feedback == 0:
            feedback_sure = 0
            feedback_unsure = 1
            print("You chose X0")
        elif human_feedback == 2:
            if rand:
                feedback_sure = int(torch.distributions.Bernoulli(0.5).sample(torch.Size([1])).item())
            else:
                feedback_sure = 1
            feedback_unsure = 0
            print("You are unsure. We follow BO recommendation.")
        else:
            raise ValueError("You can select only 0 or 1 or 2")
        y_pairwise_next = self.tensor(1 - feedback_sure).unsqueeze(0)
        y_pairwise_unsure_next = self.tensor(feedback_unsure).unsqueeze(0)
        return y_pairwise_next, y_pairwise_unsure_next
    
    def GPSHAP_UCB(self, model, Xall, Yall, beta):
        """
        Compute the GP-SHAP values.
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - X: torch.tensor, the observed inputs
        - Y: torch.tensor, the observed outputs
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        
        Return:
        - gpucbshap: GPSHAP.acquisition_functions.ucb.GPUCBSHAP, the instansiated GPSHAP class
        """
        gp_regression = ExactGPRegression(
            model.train_inputs[0].float(),
            model.train_targets.float(),
            kernel=RBFKernel,
        )
        gp_regression.fit(learning_rate=1e-2, training_iteration=100)

        # Instantiate your GPSHAP first.
        gpshap = GPSHAP(
            train_X=Xall.float(),
            scale=Yall.std().numpy(),
            model=gp_regression,
            kernel=RBFKernel(),
            include_likelihood_noise_for_explanation=False,
        )
        gpucbshap = GPUCBSHAP(
            gpshap=gpshap,
            beta_reg=1.96,
        )
        return gpucbshap
    
    def shapley_explanation(self, X_suggest, Xall, Yall, model, beta):
        """
        Visualise the GP-SHAP explanation.
        
        Args:
        - X_suggest: torch.tensor, a pairwise candidate
        - Xall: torch.tensor, the observed inputs
        - Yall: torch.tensor, the observed outputs
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        """
        # 1. learn Shapley values
        gpucbshap = self.GPSHAP_UCB(model, Xall, Yall, beta)
        gpucbshap.fit_ucbshap(X_suggest.float(), num_coalitions=2**Xall.shape[-1])
        ucb_shapley = gpucbshap.ucb_explanations
        mean_shapley = gpucbshap.mean_svs
        std_shapley = gpucbshap.std_svs

        # 2. visualise
        data_ids = ["X0", "X1"]
        if self.feature_names == None:
            feature_names = ["SUM"] + ["dim"+str(i) for i in range(Xall.shape[-1])]
        else:
            feature_names = ["SUM"] + self.feature_names
        
        fig, axes = plt.subplots(2, 3, figsize=(6, 4), gridspec_kw={'wspace': 0.6}, sharex=True)

        for i, id in enumerate(data_ids):
            df_mean = pd.DataFrame(
                torch.vstack([ucb_shapley.sum(axis=0), ucb_shapley]).T, 
                index=data_ids,
                columns=feature_names,
            )
            df_mean.iloc[i,:].plot(
                kind="barh", 
                rot=0, 
                stacked=True, 
                ax = axes[i, 0], 
                legend=False,
                color=["k","b","b","b"],
            ).invert_yaxis()
            axes[i, 0].axvline(x=0, color="k", linewidth=1)
            axes[i, 0].axhline(y=0.5, color="k", linestyle="--", linewidth=1)
            axes[i, 0].set_ylabel(id)

            df_mean = pd.DataFrame(
                torch.vstack([mean_shapley.sum(axis=0), mean_shapley]).T, 
                index=data_ids, 
                columns=feature_names,
            )
            df_mean.iloc[i,:].plot(
                kind="barh", 
                rot=0, 
                stacked=True, 
                ax = axes[i, 1], 
                yticks=[], 
                ylabel='',
                color=["k","b","b","b"],
            ).invert_yaxis()
            axes[i, 1].axvline(x=0, color="k", linewidth=1)
            axes[i, 1].axhline(y=0.5, color="k", linestyle="--", linewidth=1)

            df_std = pd.DataFrame(
                torch.vstack([std_shapley.sum(axis=0), std_shapley]).T, 
                index=data_ids, 
                columns=feature_names,
            )
            df_std.iloc[i,:].plot(
                kind="barh", 
                rot=0, 
                stacked=True, 
                ax = axes[i, 2], 
                yticks=[], 
                ylabel='',
                color=["k","b","b","b"],
            ).invert_yaxis()
            axes[i, 2].axvline(x=0, color="k", linewidth=1)
            axes[i, 2].axhline(y=0.5, color="k", linestyle="--", linewidth=1)

        axes[0, 0].set_title("UCB")
        axes[0, 1].set_title("GP mean")
        axes[0, 2].set_title("GP stddev")
        plt.show()

    def pointwise_explanation(self, X_suggest, model, beta):
        """
        Visualise the UCB, predictive mean and standard deviation.
        
        Args:
        - X_suggest: torch.tensor, a pairwise candidate
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        """
        # 1. Compute values
        with torch.no_grad():
            predictive_dist = model.likelihood(model(X_suggest))
        mean = predictive_dist.loc
        std = predictive_dist.variance.sqrt()
        ucb = mean + beta * std

        # 2. visualise
        data_ids = ["X0", "X1"]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 4), gridspec_kw={'wspace': 0.2})
        ax1.barh(data_ids, ucb, alpha=.5, color="blue")
        ax1.invert_yaxis()
        ax1.set_title("GP-UCB")

        ax2.barh(data_ids, mean, alpha=.5, color="blue")
        ax2.invert_yaxis()
        ax2.set_title("GP predictive mean")

        ax3.barh(data_ids, std, alpha=.5, color="blue")
        ax3.invert_yaxis()
        ax3.set_title("GP predictive std")

        #_ = ax3.legend(bbox_to_anchor=(1.0, 1.02), loc='upper left')
        plt.plot()
        plt.show()
        
    def visualisation_flow(self, X_suggest, Xall, Yall, model, prior_pref, beta, pref=False):
        """
        The flow for the spatial visualisation.
        
        Args:
        - X_suggest: torch.tensor, a pairwise candidate
        - Xall: torch.tensor, the observed inputs
        - Yall: torch.tensor, the observed outputs
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        - pref: bool, visualise the preference augmented model if true, otherwise the normal GP model.
        """
        print("Do you want to specify the dimension by yourself (y)? Otherwise, we estimate based on Shapley value (n)")
        self_exp = input('Type y if yes, otherwise type n as no')
        if self_exp == "y":
            wish_repeat = "y"
            while wish_repeat == "y":
                vis = HumanVisualisation(Xall, Yall, X_suggest)
                if pref:
                    vis.plot_pref(prior_pref)
                else:
                    vis.plot_gp(model)
                print("Do you want to visualise again?")
                wish_repeat = input('Type y if yes, otherwise type n as no')
        else:
            vis = ShapleyVisualisation(Xall, Yall, X_suggest)
            vis.set_plotting_range(X_suggest, model, beta)
            if pref:
                vis.plot_pref(prior_pref)
            else:
                vis.plot_gp(model)
        
    def explanation_flow(self, X_suggest, Xall, Yall, model, prior_pref, beta):
        """
        The flow for the whole explanation scheme.
        
        Args:
        - X_suggest: torch.tensor, a pairwise candidate
        - Xall: torch.tensor, the observed inputs
        - Yall: torch.tensor, the observed outputs
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        """
        print("Do you need more explanation on the two candidates?")
        value_exp = input('Type y if yes, otherwise type n as no')
        if value_exp == "y":
            self.pointwise_explanation(X_suggest, model, beta)

        print("Do you want to know the attribution to each feature?")
        shapley_exp = input('Type y if yes, otherwise type n as no')
        if shapley_exp == "y":
            self.shapley_explanation(X_suggest, Xall, Yall, model, beta)
            
        print("Do you want to visualise the GP on the plane where two candidates are placed?")
        vis_exp = input('Type y if yes, otherwise type n as no')
        if vis_exp == "y":
            self.visualisation_flow(X_suggest, Xall, Yall, model, prior_pref, beta, pref=False)
                
        print("Do you want to visualise the learnt preference model on the plane where two candidates are placed?")
        pref_exp = input('Type y if yes, otherwise type n as no')
        if pref_exp == "y":
            self.visualisation_flow(X_suggest, Xall, Yall, model, prior_pref, beta, pref=True)

            
class ShapleyVisualisation(TensorManager):
    def __init__(self, X, Y, X_pairwise_next, resolution=50):
        """
        A class for spatial visualisation based on Shapley values.
        
        Args:
        - X: torch.tensor, the observed inputs
        - Y: torch.tensor, the observed outputs
        - X_pairwise_next: torch.tensor, a pairwise candidate
        - resolution: int, the resolution for the 2D visualisation
        """
        TensorManager.__init__(self)
        self.X_best = X[Y.argmax()]
        self.n_dims = len(self.X_best)
        self.X = X
        self.Y = Y
        self.X_pref, self.X_bo = torch.chunk(X_pairwise_next.view(1,-1), dim=1, chunks=2)
        self.resolution = resolution
        
    def GPSHAP_UCB(self, model, beta):
        """
        The flow for the whole explanation scheme.
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        
        Return:
        - gpucbshap: GPSHAP.acquisition_functions.ucb.GPUCBSHAP, the instansiated GPSHAP class
        """
        gp_regression = ExactGPRegression(
            model.train_inputs[0].float(),
            model.train_targets.float(),
            kernel=RBFKernel,
        )
        gp_regression.fit(learning_rate=1e-2, training_iteration=100)

        # Instantiate your GPSHAP first.
        gpshap = GPSHAP(
            train_X=self.X.float(),
            scale=self.Y.std().numpy(),
            model=gp_regression,
            kernel=RBFKernel(),
            include_likelihood_noise_for_explanation=False,
        )
        gpucbshap = GPUCBSHAP(
            gpshap=gpshap,
            beta_reg=1.96,
        )
        return gpucbshap
        
    def set_plotting_range(self, X_pairwise_next, model, beta):
        """
        Set up the plotting range.
        
        Args:
        - X_pairwise_next: torch.tensor, a pairwise candidate
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        - beta: float, optimization hyperparameter of GP-UCB, UCB := mu(x) + beta * stddev(x)
        """
        # compute Shapley values
        gpucbshap = self.GPSHAP_UCB(model, beta)
        gpucbshap.fit_ucbshap(X_pairwise_next.float(), num_coalitions=2**self.X.shape[-1])
        ucb_shapley = gpucbshap.ucb_explanations
        
        # compute dims and bounds
        self.dims = ucb_shapley.mean(axis=0).sort(descending=True).indices[:2]
        Xall = torch.vstack([self.X_best[self.dims], self.X_bo[0,self.dims], self.X_pref[0,self.dims]])
        mins = Xall.min(axis=0).values
        maxs = Xall.max(axis=0).values
        deltas = maxs - mins
        ub = deltas + maxs
        lb = mins - deltas
        self.bounds = torch.vstack([lb, ub])
        self.x_grid = self.generate_grid()
    
    def set_subtract(self, A, B):
        """
        Compute the set subtract. C = A \ B
        
        Args:
        - A: torch.tensor, a large set
        - B: torch.tensor, a small set being contained by A.
        
        Return:
        - C: torch.tensor, A \ B
        """
        # assume A > B in the size
        mask = self.ones(A.shape).bool()
        mask[B] = 0
        return torch.masked_select(A, mask)
    
    def generate_grid(self):
        """
        Compute the grid.
        
        Return:
        - x_grid: torch.tensor, 2D grid inputs
        """
        vector = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * self.standardise_tensor(
            torch.linspace(0, 1, self.resolution).repeat(2,1).T
        )
        v1, v2 = torch.chunk(vector, dim=1, chunks=2)
        x_grid = torch.vstack([
            torch.hstack([v1, v.repeat(self.resolution,1)])
            for v in v2
        ])
        if self.n_dims > 2:
            dim_rest = self.set_subtract(torch.arange(self.n_dims), self.dims)
            dim_order = torch.cat([self.dims, dim_rest])
            X_rest = torch.vstack([self.X_pref, self.X_bo, self.X_best]).mean(axis=0)[dim_rest]
            x_grid = torch.hstack([x_grid, X_rest.repeat(len(x_grid), 1)])[:,dim_order]
        return x_grid
    
    def plot_function(self, data, ax):
        """
        Visualise the given data.
        
        Args:
        - data: 2D data
        - ax: matplotlib.pyplot.axes, axis to visualise
        """
        data = self.numpy(data).reshape(self.resolution, self.resolution)
        image = ax.imshow(
            data,
            cmap=plt.get_cmap("jet"),
            vmin=data.min(),
            vmax=data.max(),
            extent=[self.bounds[0,0], self.bounds[1,0], self.bounds[0,1], self.bounds[1,1]],
            interpolation='nearest',
            origin='lower',
        )
    
    def get_inbound(self):
        """
        Judge whether or not all observed points are contained in the defined bounds.
        
        Return:
        - X_inbound: torch.tensor, the observed points which is within the bounds.
        - flag: bool, true if all observed points are within the bounds, otherwise false.
        """
        idx_inbound = torch.logical_and(
            (self.X[:, self.dims] >= self.bounds[0]).all(axis=1), 
            (self.X[:, self.dims] <= self.bounds[1]).all(axis=1),
        )
        X_inbound = self.X[idx_inbound]
        flag = (not len(X_inbound) == 0)
        return X_inbound, flag
        
    def plot_mean_and_variance(self, mean, var, title_mean, title_var):
        """
        Visualise the GP mean and variance.
        
        Args:
        - mean: torch.tensor, the posterior predictive mean of the GP
        - var: torch.tensor, the posterior predictive variance of the GP
        - title_mean: string, the title of the figure visualising the GP mean.
        - title_var: string, the title of the figure visualising the GP variance.
        """
        print("best observed, white dot o:" + str(self.X_best))
        print("X0 (preference), black star *:" + str(self.X_pref.squeeze()))
        print("X1 (normal UCB), green cross +:" + str(self.X_bo.squeeze()))
        print("observed points, yellow cross x")

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(6,6), tight_layout=True)
        res = (self.sqrt(len(mean)) -1).long()
        self.plot_function(mean, ax1)
        ax1.scatter(self.X_best[self.dims[0]], self.X_best[self.dims[1]], color="white", marker="o", s=50)
        ax1.scatter(self.X_bo[0, self.dims[0]], self.X_bo[0, self.dims[1]], color="green", marker="+", s=50)
        ax1.scatter(self.X_pref[0, self.dims[0]], self.X_pref[0, self.dims[1]], color="black", marker="*", s=50)
        X_inbound, flag = self.get_inbound()
        if flag:
            ax1.scatter(X_inbound[:, self.dims[0]], X_inbound[:, self.dims[1]], color="yellow", marker="x", s=10)
        
        ax1.set_title(title_mean)
        ax1.set_xlabel("dimension "+str(self.dims[0].item()))
        ax1.set_ylabel("dimension "+str(self.dims[1].item()))

        self.plot_function(var, ax2)
        ax2.scatter(self.X_best[self.dims[0]], self.X_best[self.dims[1]], color="white", marker="o", s=50)
        ax2.scatter(self.X_bo[0, self.dims[0]], self.X_bo[0, self.dims[1]], color="green", marker="+", s=50)
        ax2.scatter(self.X_pref[0, self.dims[0]], self.X_pref[0, self.dims[1]], color="black", marker="*", s=50)
        if flag:
            ax2.scatter(X_inbound[:, self.dims[0]], X_inbound[:, self.dims[1]], color="yellow", marker="x", s=10)
        ax2.set_title(title_var)
        ax2.set_xlabel("dimension "+str(self.dims[0].item()))
        ax2.set_ylabel("dimension "+str(self.dims[1].item()))
        plt.show()
        
    def plot_gp(self, model):
        """
        Visualise the GP mean and variance.
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        """
        pred = predict(self.x_grid, model)
        y_mean, y_var = pred.loc, pred.variance
        self.plot_mean_and_variance(y_mean, y_var, "GP mean", "GP variance")
        
    def plot_pref(self, prior_pref):
        """
        Visualise the preference-augmented GP mean and variance.
        
        Args:
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        """
        prior_mean, prior_std = prior_pref.probability(self.x_grid, both=True)
        self.plot_mean_and_variance(prior_mean, prior_std, "Preference satisfaction", "Estimation variance")


class HumanVisualisation(TensorManager):
    def __init__(self, X, Y, X_pairwise_next, resolution=50):
        """
        A class for spatial visualisation based on a human user preference.
        
        Args:
        - X: torch.tensor, the observed inputs
        - Y: torch.tensor, the observed outputs
        - X_pairwise_next: torch.tensor, a pairwise candidate
        - resolution: int, the resolution for the 2D visualisation
        """
        TensorManager.__init__(self)
        self.X_best = X[Y.argmax()]
        self.n_dims = len(self.X_best)
        self.X = X
        self.Y = Y
        self.X_pref, self.X_bo = torch.chunk(X_pairwise_next.view(1,-1), dim=1, chunks=2)
        self.resolution = resolution
        self.dims, self.bounds = self.ask_details()
        self.x_grid = self.generate_grid() 
        
    def ask_details(self):
        """
        Ask the user to define the ranges and two dimensions to visualise
        
        Args:
        - dim: torch.tensor, the index of the two dimensions to visualise
        - bounds: torch.tensor, the bounds to visualise
        """
        dims = input('Type two dimensions you wish to see, like [0,1]')
        lb = input('Type lower bounds of each dimension you wish to see, like [-1,-1]')
        ub = input('Type upper bounds of each dimension you wish to see, like [1,1]')
        dims = self.tensor(ast.literal_eval(dims)).long()
        lb = self.tensor(ast.literal_eval(lb))
        ub = self.tensor(ast.literal_eval(ub))
        bounds = torch.vstack([lb, ub])
        return dims, bounds
    
    def set_subtract(self, A, B):
        """
        Compute the set subtract. C = A \ B
        
        Args:
        - A: torch.tensor, a large set
        - B: torch.tensor, a small set being contained by A.
        
        Return:
        - C: torch.tensor, A \ B
        """
        # assume A > B in the size
        mask = self.ones(A.shape).bool()
        mask[B] = 0
        return torch.masked_select(A, mask)
    
    def generate_grid(self):
        """
        Compute the grid.
        
        Return:
        - x_grid: torch.tensor, 2D grid inputs
        """
        vector = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * self.standardise_tensor(
            torch.linspace(0, 1, self.resolution).repeat(2,1).T
        )
        v1, v2 = torch.chunk(vector, dim=1, chunks=2)
        x_grid = torch.vstack([
            torch.hstack([v1, v.repeat(self.resolution,1)])
            for v in v2
        ])
        if self.n_dims > 2:
            dim_rest = self.set_subtract(torch.arange(self.n_dims), self.dims)
            dim_order = torch.cat([self.dims, dim_rest])
            X_rest = torch.vstack([self.X_pref, self.X_bo, self.X_best]).mean(axis=0)[dim_rest]
            x_grid = torch.hstack([x_grid, X_rest.repeat(len(x_grid), 1)])[:,dim_order]
        return x_grid
    
    def plot_function(self, data, ax):
        """
        Visualise the given data.
        
        Args:
        - data: 2D data
        - ax: matplotlib.pyplot.axes, axis to visualise
        """
        data = self.numpy(data).reshape(self.resolution, self.resolution)
        image = ax.imshow(
            data,
            cmap=plt.get_cmap("jet"),
            vmin=data.min(),
            vmax=data.max(),
            extent=[self.bounds[0,0], self.bounds[1,0], self.bounds[0,1], self.bounds[1,1]],
            interpolation='nearest',
            origin='lower',
        )
    
    def get_inbound(self):
        """
        Judge whether or not all observed points are contained in the defined bounds.
        
        Return:
        - X_inbound: torch.tensor, the observed points which is within the bounds.
        - flag: bool, true if all observed points are within the bounds, otherwise false.
        """
        idx_inbound = torch.logical_and(
            (self.X[:, self.dims] >= self.bounds[0]).all(axis=1), 
            (self.X[:, self.dims] <= self.bounds[1]).all(axis=1),
        )
        X_inbound = self.X[idx_inbound]
        flag = (not len(X_inbound) == 0)
        return X_inbound, flag
        
    def plot_mean_and_variance(self, mean, var, title_mean, title_var):
        """
        Visualise the GP mean and variance.
        
        Args:
        - mean: torch.tensor, the posterior predictive mean of the GP
        - var: torch.tensor, the posterior predictive variance of the GP
        - title_mean: string, the title of the figure visualising the GP mean.
        - title_var: string, the title of the figure visualising the GP variance.
        """
        print("best observed, white dot o:" + str(self.X_best))
        print("X0 (preference), black star *:" + str(self.X_pref.squeeze()))
        print("X1 (normal UCB), green cross +:" + str(self.X_bo.squeeze()))
        print("observed points, yellow cross x")

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(6,6), tight_layout=True)
        res = (self.sqrt(len(mean)) -1).long()
        self.plot_function(mean, ax1)
        ax1.scatter(self.X_best[self.dims[0]], self.X_best[self.dims[1]], color="white", marker="o", s=50)
        ax1.scatter(self.X_bo[0, self.dims[0]], self.X_bo[0, self.dims[1]], color="green", marker="+", s=50)
        ax1.scatter(self.X_pref[0, self.dims[0]], self.X_pref[0, self.dims[1]], color="black", marker="*", s=50)
        X_inbound, flag = self.get_inbound()
        if flag:
            ax1.scatter(X_inbound[:, self.dims[0]], X_inbound[:, self.dims[1]], color="yellow", marker="x", s=10)
        
        ax1.set_title(title_mean)
        ax1.set_xlabel("dimension "+str(self.dims[0].item()))
        ax1.set_ylabel("dimension "+str(self.dims[1].item()))

        self.plot_function(var, ax2)
        ax2.scatter(self.X_best[self.dims[0]], self.X_best[self.dims[1]], color="white", marker="o", s=50)
        ax2.scatter(self.X_bo[0, self.dims[0]], self.X_bo[0, self.dims[1]], color="green", marker="+", s=50)
        ax2.scatter(self.X_pref[0, self.dims[0]], self.X_pref[0, self.dims[1]], color="black", marker="*", s=50)
        if flag:
            ax2.scatter(X_inbound[:, self.dims[0]], X_inbound[:, self.dims[1]], color="yellow", marker="x", s=10)
        ax2.set_title(title_var)
        ax2.set_xlabel("dimension "+str(self.dims[0].item()))
        ax2.set_ylabel("dimension "+str(self.dims[1].item()))
        plt.show()
        
    def plot_gp(self, model):
        """
        Visualise the GP mean and variance.
        
        Args:
        - model: botorch.models.gp_regression.SingleTaskGP, BoTorch SingleTaskGP.
        """
        y_mean, y_var = predict(self.x_grid, model)
        self.plot_mean_and_variance(y_mean, y_var, "GP mean", "GP variance")
        
    def plot_pref(self, prior_pref):
        """
        Visualise the preference-augmented GP mean and variance.
        
        Args:
        - prior_pref: CoExBO._monte_carlo_quadrature.MonteCarloQuadrature, soft-Copleland score function (human preference).
        """
        prior_mean, prior_std = prior_pref.probability(self.x_grid, both=True)
        self.plot_mean_and_variance(prior_mean, prior_std, "Preference satisfaction", "Estimation variance")
