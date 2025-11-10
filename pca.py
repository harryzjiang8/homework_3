# YOUR IMPORTS HERE
# you may want to use `from sklearn.exceptions import NotFittedError` for the `transform` method
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

class Standard_PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.fitted = False

    def fit(self, X: np.ndarray):
        # compute and store mean and std
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # standardize X
        Z = (X - self.mean_) / self.std_

        # fit PCA
        self.pca = PCA(n_components=self.n_components, svd_solver="full")
        self.pca.fit(Z)

        self.fitted = True

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # compute and store mean and std
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # standardize X
        Z = (X - self.mean_) / self.std_

        # fit PCA and transform
        self.pca = PCA(n_components=self.n_components, svd_solver="full")
        X_pca = self.pca.fit_transform(Z)

        self.fitted = True
        return X_pca

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise NotFittedError("PCA has not been fitted")

        # standardize X using stored parameters
        Z = (X - self.mean_) / self.std_

        # project onto PCA components
        return self.pca.transform(Z)


def PCA_dim(X: np.ndarray, p: float) -> int:
    # create Standard_PCA object
    n_components = X.shape[1]
    pca = Standard_PCA(n_components)
    pca.fit(X)
    # get the variance ratios and get the cumulative sum
    var_ratios = pca.pca.explained_variance_ratio_.cumsum()
    
    # go through the ratios until it is over given p
    for i in range(len(var_ratios)):
        if p <= var_ratios[i]:
            return i+1
    pass
