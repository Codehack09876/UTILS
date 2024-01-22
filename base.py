"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

         self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Recursive function to build the decision tree

        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)  # Leaf node, return the mean value for regression

        # Find the optimal split attribute and value
        split_attr, split_value = opt_split_attribute(X, y, self.criterion, X.columns)

        # Split the data
        X_left, y_left, X_right, y_right = split_data(X, y, split_attr, split_value)

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return (split_attr, split_value, left_subtree, right_subtree)

        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

         return pd.Series([self._predict_single(x, self.tree) for _, x in X.iterrows()], index=X.index)

    def _predict_single(self, x, node):
        if isinstance(node, (int, float)):
            return node  # Leaf node, return the mean value for regression

        split_attr, split_value, left_subtree, right_subtree = node

        if x[split_attr] <= split_value:
            return self._predict_single(x, left_subtree)
        else:
            return self._predict_single(x, right_subtree)

        pass

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

        self._plot_tree(self.tree)

    def _plot_tree(self, node, depth=0):
        if isinstance(node, (int, float)):
            print(f"Y: {node}")
            return

        split_attr, split_value, left_subtree, right_subtree = node

        print(f"?(X{split_attr} > {split_value})")
        print("  " * depth + "Y:", end=" ")
        self._plot_tree(left_subtree, depth + 1)
        print("  " * depth + "N:", end=" ")
        self._plot_tree(right_subtree, depth+1)

        
        pass
