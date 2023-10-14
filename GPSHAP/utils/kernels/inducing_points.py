import torch
from sklearn.cluster import KMeans
from torch import FloatTensor

"""
def compute_inducing_points(train_X: FloatTensor, num_inducing_points: int) -> FloatTensor:
    selecting inducing points using k-means algorithm if number of inducing points != training points
    

    if num_inducing_points == train_X.shape[0]:
        return train_X

    kmeans = KMeans(n_clusters=num_inducing_points)
    kmeans.fit(train_X.numpy())

    return torch.tensor(kmeans.cluster_centers_).float()
"""

def compute_inducing_points(train_X: FloatTensor, num_inducing_points: int) -> FloatTensor:
    """selecting inducing points using k-means algorithm if number of inducing points != training points
    """

    if num_inducing_points == train_X.shape[0]:
        return train_X

    mins = train_X.min(axis=0).values
    maxes = train_X.max(axis=0).values
    X_base = torch.linspace(0, 1, num_inducing_points).repeat(train_X.shape[1], 1).T
    X_equispaced = X_base * (maxes - mins) + mins

    return X_equispaced