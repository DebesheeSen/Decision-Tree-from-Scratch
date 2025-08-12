# decision_tree.py
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_index=None, branches=None, value=None):
        self.feature_index = feature_index  # which column to split on
        self.branches = branches or {}      # dict: category_value → Node
        self.value = value                  # label if leaf
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5, criterion='gini', n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_features = n_features
        self.root = None
    
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self.build_tree(X, y)
    
    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique = np.unique(y)
        if(depth > self.max_depth or len(unique)==1 or n_samples < self.min_samples_split):
            val = self.most_common_label(y)
            return Node(value = val)
        if self.n_features is None:
            feature_indexes = np.arange(n_features)
        else:
            feature_indexes = np.random.choice(n_features, self.n_features, replace=False)
        best_feature = self.best_split(X, y, feature_indexes)
        branches = {}
        feature_values = np.unique(X[:, best_feature])
    
        for val in feature_values:
            subset_mask = X[:, best_feature] == val
            X_subset = X[subset_mask]
            y_subset = y[subset_mask]
            
            if len(y_subset) == 0:
                branches[val] = Node(value=self.most_common_label(y))
            else:
                branches[val] = self.build_tree(X_subset, y_subset, depth + 1)
    
        return Node(feature_index=best_feature, branches=branches)
    
    def best_split(self, X, y, feature_indexes):
        best_gain = -1
        split_idx = None
        for f in feature_indexes:
            if(self.criterion == 'gini'):
                gain = self.gini(X[:,f], y)
            else:
                gain = self.info_gain(X[:,f], y)
            
            if gain > best_gain:
                best_gain = gain
                split_idx = f
        return split_idx
        
    def entropy(self, y):
        ps = np.bincount(y)/len(y)
        ent = -np.sum(p*np.log2(p) for p in ps if p > 0)
        return ent
        
    def info_gain(self, X_col, y):
        parent_entropy = self.entropy(y)
        
        values = np.unique(X_col)
        weighted_entropy = 0
        for val in values:
            subset_mask = X_col == val
            weighted_entropy += (np.sum(subset_mask) / len(y)) * self.entropy(y[subset_mask])
        return parent_entropy - weighted_entropy
    
    def gini(self, X_col, y):
        parent_gini = self.gini_impurity(y)
    
        values = np.unique(X_col)
        weighted_gini = 0
        for val in values:
            subset_mask = X_col == val
            weighted_gini += (np.sum(subset_mask) / len(y)) * self.gini_impurity(y[subset_mask])
    
        return parent_gini - weighted_gini
    
    def gini_impurity(self, y):
        probs = [np.sum(y == i)/len(y) for i in np.unique(y)]
        return (1 - np.sum(np.square(probs)))
    
    def most_common_label(self, y):
        most_common_value = Counter(y).most_common(1)[0][0]
        return most_common_value
    
    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])
    
    def traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        feature_val = x[node.feature_index]
        if feature_val in node.branches:
            return self.traverse_tree(x, node.branches[feature_val])
        else:
            leaf_values = [branch.value for branch in node.branches.values() if branch.is_leaf_node()]
            if leaf_values:
                return self.most_common_label(np.array(leaf_values))
            else:
                return self.most_common_label(np.array([0, 1]))  # fallback