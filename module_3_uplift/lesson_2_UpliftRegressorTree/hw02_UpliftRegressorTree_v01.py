import numpy as np
from typing import Iterable

class UpliftTreeRegressor:
    
    ## init class instance
    def __init__(self,
        max_depth: int = 3, # максимальная глубина дерева.
        min_samples_leaf: int = 1000, # минимальное необходимое число обучающих объектов в листе дерева.
        min_samples_leaf_treated: int = 300, # минимальное необходимое число обучающих объектов с T=1 в листе дерева.
        min_samples_leaf_control: int = 300, # минимальное необходимое число обучающих объектов с T=0 в листе дерева.
    ):
        
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        
    # fit the model
    def fit(
        self,
        X: np.ndarray, # массив (n * k) с признаками.
        treatment: np.ndarray, # массив (n) с флагом воздействия.
        y: np.ndarray # массив (n) с целевой переменной.
    ) -> None:
        
        self.n_features_ = X.shape[1]
        self.tree = self._grow_tree(X, treatment, y)
    
    
    def _grow_tree(self, X, treatment, y, depth=0):
        
        predicted_ate = self._ATE(treatment, y)
        
        node = Node(
            n_items=y.size,
            ATE=predicted_ate,
        )
        
         # Split recursively until maximum depth is reached
        if depth < self.max_depth:
            idx, thr = self._best_split(X, treatment, y)
            
            if idx is not None:
                ## split data by threshold
                indices_left = X[:, idx] <= thr
                X_left, treatment_left, y_left = \
                    X[indices_left], treatment[indices_left], y[indices_left]
                X_right, treatment_right, y_right = \
                    X[~indices_left], treatment[~indices_left], y[~indices_left]
                
                node.split_feat = idx
                node.split_threshold = thr
                node.left = self._grow_tree(X_left, treatment_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, treatment_right, y_right, depth + 1)
        return node
    
    def _ATE(self, treatment, y):
        
        return (treatment * y).sum() / treatment.sum() - \
            ((1 - treatment) * y).sum() / (1 - treatment).sum()
        
    def _best_split(self, X, treatment, y):
        
        if (y.size < self.min_samples_leaf * 2) or \
            (np.sum(treatment) < self.min_samples_leaf_treated) or \
            (np.sum(1 - treatment) < self.min_samples_leaf_control):
            
            return None, None
        
        best_ddp = 0
        best_idx, best_thr = None, None
        
         # Loop through all features
        for idx in range(X.shape[1]):
            
            unique_values = np.unique(X[:, idx])
            if len(unique_values) > 10:
                percentiles = np.percentile(X[:, idx], [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
            else:
                percentiles = np.percentile(unique_values, [10, 50, 90])
                
            threshold_options = np.unique(percentiles)
            
                      
            for thr_i in threshold_options:  # possible split positions
                
                indices_left = X[:, idx] <= thr_i
                treatment_left, y_left = \
                    treatment[indices_left], y[indices_left]
                treatment_right, y_right = \
                    treatment[~indices_left], y[~indices_left]
                
                ate_left = self._ATE(treatment_left, y_left)
                ate_right = self._ATE(treatment_right, y_right)
                
                ddp = np.abs(ate_right - ate_left)

                if (ddp > best_ddp) and \
                    (y_left.size >= self.min_samples_leaf) and \
                    (np.sum(treatment_left) >= self.min_samples_leaf_treated) and \
                    (np.sum(1 - treatment_left) >= self.min_samples_leaf_control) and \
                    (y_right.size >= self.min_samples_leaf) and \
                    (np.sum(treatment_right) >= self.min_samples_leaf_treated) and \
                    (np.sum(1 - treatment_right) >= self.min_samples_leaf_control):
                    
                    best_ddp = ddp
                    best_idx = idx
                    best_thr = thr_i

        return best_idx, best_thr
        
        
    
    def predict(self, X: np.ndarray) -> Iterable[float]:
        # compute predictions
        # pass
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree
        while node.left:
            if inputs[node.split_feat] < node.split_threshold:
#                 print("node.left ATE {}".format(node.ATE))
                node = node.left
            else:
#                 print("node.right ATE {}".format(node.ATE))
                node = node.right
        return node.ATE
    

class Node:
    def __init__(self, n_items, ATE):
        
        self.n_items = n_items
        self.ATE = ATE
        self.split_feat = None
        self.split_threshold = None
        self.left = None
        self.right = None
        