import torch

from dataclasses import dataclass, field
from torch import FloatTensor, Tensor

from GPSHAP.explanation_algorithms.GPSHAP import GPSHAP

kw_only=True

@dataclass#(kw_only=True)
class GPUCBSHAP(object):
    """Gaussian process upper confidence bound SHAP

    To explain the acquisition function

    Parameters
    ----------
    gpshap: instantiated GPSHAP class
    beta_reg: the trade-off parameter between exploitation and exploration.
    """

    gpshap: GPSHAP
    beta_reg: FloatTensor

    mean_svs: Tensor = field(init=False)
    std_svs: Tensor = field(init=False)
    covar_svs: Tensor = field(init=False)
    ucb_explanations: Tensor = field(init=False)

    def fit_ucbshap(self, X: FloatTensor, num_coalitions: int) -> None:
        self.gpshap.fit_gpshap(X, num_coalitions)
        self.mean_svs = self.gpshap.return_mean_stochastic_shapley_values()
        self.covar_svs = self.gpshap.return_gpshap_uncertainties_for_each_query()

        # from covar_svs extract variance for each X.
        self.std_svs = torch.stack([
            self.covar_svs[:, :, i].diag().sqrt() for i in range(X.shape[0])
        ]).T

        self.ucb_explanations = self.mean_svs + self.beta_reg * self.std_svs

        return None
