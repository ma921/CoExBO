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
from ._gp import predict

class HumanFeedback(TensorManager):
    def __init__(self):
        TensorManager.__init__(self)
        
    def display_pairwise_samples(self, X_pairwise_next, random=False):
        X0, X1 = torch.chunk(X_pairwise_next, dim=1, chunks=2)
        if random:
            print("X1 (random): " + str(X1.squeeze().float()))
            print("X0 (random): " + str(X0.squeeze().float()))
        else:
            print("X1 (normal UCB): " + str(X1.squeeze().float()))
            print("X0 (preference): " + str(X0.squeeze().float()))
        
    def get_human_feedback(self, rand=False):
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
        # 1. learn Shapley values
        gpucbshap = self.GPSHAP_UCB(model, Xall, Yall, beta)
        gpucbshap.fit_ucbshap(X_suggest.float(), num_coalitions=2**Xall.shape[-1])
        ucb_shapley = gpucbshap.ucb_explanations
        mean_shapley = gpucbshap.mean_svs
        std_shapley = gpucbshap.std_svs

        # 2. visualise
        data_ids = ["X0", "X1"]
        feature_names = ["dim"+str(i) for i in range(Xall.shape[-1])]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), gridspec_kw={'wspace': 0.2})
        df_mean = pd.DataFrame(ucb_shapley.T, index=data_ids, columns=feature_names)
        df_mean.plot(kind="barh", rot=0, stacked=True, ax = ax1, legend=False)
        ax1.set_xlabel("UCB Shapley values")
        ax1.set_title("UCB")

        df_mean = pd.DataFrame(mean_shapley.T, index=data_ids, columns=feature_names)
        df_mean.plot(kind="barh", rot=0, stacked=True, ax = ax2, legend=False)
        ax2.set_xlabel("GP mean Shapley values")
        ax2.set_title("Mean")

        df_std = pd.DataFrame(std_shapley.T, index=data_ids, columns=feature_names)
        _ = df_std.plot(kind="barh", rot=0, stacked=True, ax = ax3, legend=False)
        ax3.set_xlabel("GP std Shapley values")
        ax3.set_title("Uncertainty")

        _ = ax3.legend(bbox_to_anchor=(1.0, 1.02), loc='upper left')
        plt.plot()
        plt.show()

    def pointwise_explanation(self, X_suggest, model, beta):
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
        ax1.set_xlabel("GP-UCB")
        ax1.set_title("UCB")

        ax2.barh(data_ids, mean, alpha=.5, color="blue")
        ax2.set_xlabel("GP predictive mean")
        ax2.set_title("Mean")

        ax3.barh(data_ids, std, alpha=.5, color="blue")
        ax3.set_xlabel("GP predictive std")
        ax3.set_title("Uncertainty")

        #_ = ax3.legend(bbox_to_anchor=(1.0, 1.02), loc='upper left')
        plt.plot()
        plt.show()
        
    def visualisation_flow(self, X_suggest, Xall, Yall, model, prior_pref, beta, pref=False):
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
        TensorManager.__init__(self)
        self.X_best = X[Y.argmax()]
        self.n_dims = len(self.X_best)
        self.X = X
        self.Y = Y
        self.X_pref, self.X_bo = torch.chunk(X_pairwise_next.view(1,-1), dim=1, chunks=2)
        self.resolution = resolution
        
    def GPSHAP_UCB(self, model, beta):
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
    
    def set_substract(self, A, B):
        # assume A > B in the size
        mask = self.ones(A.shape).bool()
        mask[B] = 0
        return torch.masked_select(A, mask)
    
    def generate_grid(self):
        vector = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * self.standardise_tensor(
            torch.linspace(0, 1, self.resolution).repeat(2,1).T
        )
        v1, v2 = torch.chunk(vector, dim=1, chunks=2)
        x_grid = torch.vstack([
            torch.hstack([v1, v.repeat(self.resolution,1)])
            for v in v2
        ])
        if self.n_dims > 2:
            dim_rest = self.set_substract(torch.arange(self.n_dims), self.dims)
            dim_order = torch.cat([self.dims, dim_rest])
            X_rest = torch.vstack([self.X_pref, self.X_bo, self.X_best]).mean(axis=0)[dim_rest]
            x_grid = torch.hstack([x_grid, X_rest.repeat(len(x_grid), 1)])[:,dim_order]
        return x_grid
    
    def plot_function(self, data, ax):
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
        idx_inbound = torch.logical_and(
            (self.X[:, self.dims] >= self.bounds[0]).all(axis=1), 
            (self.X[:, self.dims] <= self.bounds[1]).all(axis=1),
        )
        X_inbound = self.X[idx_inbound]
        flag = (not len(X_inbound) == 0)
        return X_inbound, flag
        
    def plot_mean_and_variance(self, mean, var, title_mean, title_var):
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
        y_mean, y_var = predict(self.x_grid, model)
        self.plot_mean_and_variance(y_mean, y_var, "GP mean", "GP variance")
        
    def plot_pref(self, prior_pref):
        prior_mean, prior_std = prior_pref.probability(self.x_grid, both=True)
        self.plot_mean_and_variance(prior_mean, prior_std, "Preference satisfaction", "Estimation variance")

class HumanVisualisation(TensorManager):
    def __init__(self, X, Y, X_pairwise_next, resolution=50):
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
        dims = input('Type two dimensions you wish to see, like [0,1]')
        lb = input('Type lower bounds of each dimension you wish to see, like [-1,-1]')
        ub = input('Type upper bounds of each dimension you wish to see, like [1,1]')
        dims = self.tensor(ast.literal_eval(dims)).long()
        lb = self.tensor(ast.literal_eval(lb))
        ub = self.tensor(ast.literal_eval(ub))
        bounds = torch.vstack([lb, ub])
        return dims, bounds
    
    def set_substract(self, A, B):
        # assume A > B in the size
        mask = self.ones(A.shape).bool()
        mask[B] = 0
        return torch.masked_select(A, mask)
    
    def generate_grid(self):
        vector = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * self.standardise_tensor(
            torch.linspace(0, 1, self.resolution).repeat(2,1).T
        )
        v1, v2 = torch.chunk(vector, dim=1, chunks=2)
        x_grid = torch.vstack([
            torch.hstack([v1, v.repeat(self.resolution,1)])
            for v in v2
        ])
        if self.n_dims > 2:
            dim_rest = self.set_substract(torch.arange(self.n_dims), self.dims)
            dim_order = torch.cat([self.dims, dim_rest])
            X_rest = torch.vstack([self.X_pref, self.X_bo, self.X_best]).mean(axis=0)[dim_rest]
            x_grid = torch.hstack([x_grid, X_rest.repeat(len(x_grid), 1)])[:,dim_order]
        return x_grid
    
    def plot_function(self, data, ax):
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
        idx_inbound = torch.logical_and(
            (self.X[:, self.dims] >= self.bounds[0]).all(axis=1), 
            (self.X[:, self.dims] <= self.bounds[1]).all(axis=1),
        )
        X_inbound = self.X[idx_inbound]
        flag = (not len(X_inbound) == 0)
        return X_inbound, flag
        
    def plot_mean_and_variance(self, mean, var, title_mean, title_var):
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
        y_mean, y_var = predict(self.x_grid, model)
        self.plot_mean_and_variance(y_mean, y_var, "GP mean", "GP variance")
        
    def plot_pref(self, prior_pref):
        prior_mean, prior_std = prior_pref.probability(self.x_grid, both=True)
        self.plot_mean_and_variance(prior_mean, prior_std, "Preference satisfaction", "Estimation variance")

            
class RhombusVisualisation(TensorManager):
    def __init__(self, X, Y, X_pairwise_next, resolution=50):
        TensorManager.__init__(self)
        self.X_best = X[Y.argmax()]
        self.Y = Y
        self.X_pref, self.X_bo = torch.chunk(X_pairwise_next.view(1,-1), dim=1, chunks=2)
        self.resolution = resolution
        self.generate_grid()        
        
    def generate_grid(self):
        v1 = torch.vstack([
            (1 - t) * self.X_best.squeeze() + t * self.X_pref.squeeze() for t in torch.linspace(0,1,self.resolution)
        ])
        v2 = torch.vstack([
            (1 - t) * self.X_best.squeeze() + t * self.X_bo.squeeze() for t in torch.linspace(0,1,self.resolution)
        ])
        diff = v2 - self.X_best
        self.x_grid = torch.vstack([
            d + v1 for d in diff
        ])
        
    def plot_function(self, data, ax):
        data = self.numpy(data).reshape(self.resolution, self.resolution)
        image = ax.imshow(
            data,
            cmap=plt.get_cmap("jet"),
            vmin=data.min(),
            vmax=data.max(),
            interpolation='nearest',
            origin='lower',
        )
    
    def plot_mean_and_variance(self, mean, var, title_mean, title_var):
        print("best observed :" + str(self.X_best))
        print("X0 (preference) :" + str(self.X_pref.squeeze()))
        print("X1 (normal UCB) :" + str(self.X_bo.squeeze()))
        
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(6,6))
        ax1.axis('off')
        res = (self.sqrt(len(mean)) -1).long()
        self.plot_function(mean, ax1)
        ax1.annotate('best observation', xy=(0, 0), xytext=(10, 10), color="black", bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=1), arrowprops=dict(facecolor='white', shrink=0.05))
        ax1.annotate('normal UCB', xy=(0, res), xytext=(10, res - 10), color="black", bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=1), arrowprops=dict(facecolor='white', shrink=0.05))
        ax1.annotate('preference', xy=(res, 0), xytext=(res - 20, 20), color="black", bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=1), arrowprops=dict(facecolor='white', shrink=0.05))
        ax1.set_title(title_mean)

        self.plot_function(var, ax2)
        ax2.axis('off')
        ax2.annotate('best observation', xy=(0, 0), xytext=(10, 10), color="black", bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=1), arrowprops=dict(facecolor='white', shrink=0.05))
        ax2.annotate('normal UCB', xy=(0, res), xytext=(10, res - 10), color="black", bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=1), arrowprops=dict(facecolor='white', shrink=0.05))
        ax2.annotate('preference', xy=(res, 0), xytext=(res - 20, 20), color="black", bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=1), arrowprops=dict(facecolor='white', shrink=0.05))
        ax2.set_title(title_var)
        plt.show()
        
    def plot_gp(self, model):
        y_mean, y_var = predict(self.x_grid, model)
        self.plot_mean_and_variance(y_mean, y_var, "GP mean", "GP variance")
        
    def plot_pref(self, prior_pref):
        prior_mean, prior_std = prior_pref.probability(self.x_grid, both=True)
        self.plot_mean_and_variance(prior_mean, prior_std, "Preference satisfaction", "Estimation variance")