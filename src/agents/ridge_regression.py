import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)
from reco_gym import Configuration

ridge_args = {
    'random_seed': np.random.randint(2 ** 31 - 1),
    'subsample_negatives': 1.,
    'mode': 'mean', # One of {'mean', 'LCB', 'UCB'}
    'alpha': 0, # alpha * variance for LCB                
    'l2': True, # Regularisation strength, grid search if True, ordinary least-squares if False
    'W': 1,
    'pid': 0
}

from numba import jit
from tqdm import tqdm, trange

@jit(nopython=True)
def act_linear_model(X, W, P):
    return np.argmax(X.dot(W.T).reshape(P))

@jit(nopython=True)
def act_with_bounds(X, W, cov, P, alpha, mode):
    # Compute standard scores
    scores = X.dot(W.T)
    # If we're in a LCB/UCB type setting, compute predictive variance
    if mode in ['LCB', 'UCB'] and alpha:
        # Compute variance for every action
        var = np.zeros(P).astype(np.float32)
        for a in np.arange(P):
            var[a] = np.sqrt(X @ cov[a] @ X.T)
        if mode == 'LCB':
            scores -= alpha * var
        elif mode == 'UCB':
            scores += alpha * var
    top_action = np.argmax(scores)
    return top_action, scores[top_action]

class RidgeDMModelBuilder(AbstractFeatureProvider):
    ''' Build a reward estimator for the Direct Method with a Ridge Regressor '''
    def __init__(self, config):
        super(RidgeDMModelBuilder, self).__init__(config)
        self.pid = 0

    def build(self):
        class RidgeDMFeaturesProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(RidgeDMFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class RidgeDMModel(Model):
            def __init__(self, config, weights, covariance, mode = self.config.mode, alpha = self.config.alpha):
                super(RidgeDMModel, self).__init__(config)
                self.W = weights 
                self.cov = covariance
                self.mode = mode
                self.alpha = alpha

            def act(self, observation, features):
                # Take argmax action
                P = features.shape[1]
                X = features.astype(np.float32)
                # Normalise
                X /= X.sum()
                # Add an extra fixed feature that corresponds to a bias term
                X = np.append(X, 1.0).astype(np.float32)

                action, reward_estimate = act_with_bounds(X.ravel(), self.W, self.cov, P, self.alpha, self.mode)

                ps_all = np.zeros(P)
                ps_all[action] = 1.0

                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': 1.0,
                        'ps-a': ps_all,
                        'r-est': reward_estimate,
                    },
                }

        # Get data
        features, actions, deltas, pss = self.train_data()
        
        X = features
        N = X.shape[0]
        P = X.shape[1]
        A = actions
        y = deltas

        X = normalize(X, axis=1, norm='l1')
        # Add an extra fixed feature that corresponds to a bias term
        X = np.hstack((X, np.ones(X.shape[0]).reshape(-1,1)))

        # Train-test split
        X_train, X_test, A_train, A_test, y_train, y_test = train_test_split(X, A, y, test_size = .2, random_state = 42)

        # GRID SEARCH
        lambdas = [.01, .1, .5, 1., 2.5, 5., 7.5, 10., 12.5, 15.] if self.config.l2 else [1e-8] # Very small, but not explicitly zero to avoid singular matrices

        # Validation metric - MSE
        MSE = []
        
        # For every value to check
        for lambda_reg in lambdas:
            # Placeholder for the predicted number of clicks
            SSE = 0
            # For every action
            with tqdm(total=P, desc='Grid {0}, lambda = {1}'.format(self.pid, lambda_reg)) as pbar:
                for action in range(P):
                    # Mask out actions for this specific item
                    mask = A_train == action
                    X_train_subset = X_train[mask]
                    y_train_subset = y_train[mask]

                    # Fit model
                    Ainv = np.linalg.inv(X_train_subset.T @ X_train_subset + lambda_reg * np.eye(P + 1))
                    model = Ainv @ X_train_subset.T @ y_train_subset

                    # Evaluate model on test set MSE
                    test_mask = A_test == action
                    X_test_subset = X_test[test_mask]
                    y_test_subset = y_test[test_mask]

                    # If we have samples for this action
                    if len(y_test_subset):
                        # Get reward estimates for every test sample
                        y_test_subset_prediction = X_test_subset @ model.T

                        # Compute and store sum of squared errors
                        SSE += np.sum((y_test_subset - y_test_subset_prediction)**2)

                    # Update progress bar
                    pbar.update(1)

            MSE.append(SSE/len(y_test))
        
        optimal_lambda = lambdas[np.argmin(MSE)]

        # Now fit model on all data with optimal lambda
        E_ctr = np.mean(y)
        P_clicks = 0

        thetas = []
        covars = []
        # For every action
        with tqdm(total=P, desc='Train {0}'.format(self.pid)) as pbar:
            for action in range(P):
                # Mask out actions for this specific item
                mask = A == action
                X_subset = X[mask]
                y_subset = y[mask]

                # Fit model
                Ainv = np.linalg.inv(X_subset.T @ X_subset + optimal_lambda * np.eye(P + 1))
                model = Ainv @ X_subset.T @ y_subset

                # Store predicted number of clicks
                P_clicks += (X_subset @ model.T).sum()

                # Store model
                thetas.append(model)
                covars.append(Ainv)
                # Update progress bar
                pbar.update(1)
        print('E-CTR: ', E_ctr, 'P-CTR: ', P_clicks/len(y), 'Bias: ', P_clicks/len(y) - E_ctr)

        # Stack model parameters and store covariance estimates
        thetas = np.vstack(thetas)
        covars = np.asarray(covars)        

        return (
            RidgeDMFeaturesProvider(self.config),
            RidgeDMModel(self.config, thetas.astype(np.float32), covars.astype(np.float32))
        )
            
class RidgeDMAgent(ModelBasedAgent):
    """
    Ridge regression Agent.
    """

    def __init__(self, config = Configuration(ridge_args)):
        super(RidgeDMAgent, self).__init__(
            config,
            RidgeDMModelBuilder(config)
        )
