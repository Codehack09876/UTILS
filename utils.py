"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import math
import numpy as np


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    is_real = all(isinstance(value, (int, float)) for value in y)
    return is_real
    pass

def mse


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    total_samples = len(Y)
    value_counts = Y.value_counts()
    entropy_val = 0
    for count in value_counts:
        probability = count / total_samples
        entropy_val -= probability * math.log2(probability)
    return entropy_val

    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    total_samples = len(Y)
    value_counts = Y.value_counts()
    gini_index_val = 0
    for count in value_counts:
        probability = count / total_samples
        gini_index_val += probability**2
    return 1-gini_index_val

    pass


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    original_entropy = entropy(Y)

    # Combine the target variable (Y) and the attribute (attr) into a DataFrame
    data = pd.DataFrame({'Y': Y, 'Attr': attr})

    # Calculate the weighted average entropy after splitting by the attribute
    split_entropy = data.groupby('Attr')['Y'].apply(entropy)
    weighted_entropy = (split_entropy * data['Attr'].value_counts() / len(data)).sum()

    # Calculate information gain
    info_gain = original_entropy - weighted_entropy

    return info_gain

    pass


def opt_split_attribute_discrete(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """


    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_attribute = None
    best_score = -np.inf if criterion == 'entropy' else np.inf

    for attribute in features:
        score = information_gain(y, X[attribute])
        if (score > best_score):
            best_score = score
            best_attribute = attribute

    return best_attribute
    pass

def opt_split_attribute_real(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    best_attribute = None
    best_value = None
    best_score = -np.inf if criterion == 'entropy' else np.inf

    for attribute in features:
        unique_values = np.unique(X[attribute])
        
        for value in unique_values:
            X_left, y_left, X_right, y_right = split_data(X, y, attribute, value)

            if criterion == 'entropy':
                score = information_gain(y, X[attribute])
            elif criterion == 'gini_index':
                score = gini_index(y)
            else:
                raise ValueError("Invalid criterion. Use 'entropy' or 'gini_index'.")

            if (score > best_score):
                best_score = score
                best_attribute = attribute
                best_value = value

    return best_attribute, best_value
    
    pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    mask = X[attribute] <= value
    return X[mask], y[mask], X[~mask], y[~mask]
    pass
