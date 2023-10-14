from botorch.acquisition.analytic import UpperConfidenceBound
from ._utils import TensorManager
tm = TensorManager()

class AcquisitionFunction(UpperConfidenceBound):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        return super().forward(x.unsqueeze(1)).detach()

def compute_distance(X):
    ground_truth = tm.zeros(X.shape[1])
    euclidean = (X - ground_truth).pow(2).sqrt().sum(axis=1)
    return -euclidean

def sort_suggestion(X_suggest, model, weights=None, label="oracle"):
    if label == "oracle":
        w_suggest = compute_distance(X_suggest)
    elif label == "ucb":
        UCB = AcquisitionFunction(model, beta=0.2)
        w_suggest = UCB(X_suggest)
    elif label == "weights":
        w_suggest = weights
    
    indices_sort = w_suggest.argsort(descending=True)
    return X_suggest[indices_sort]