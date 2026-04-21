import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def prepare_data(df, numerical_features, target_col):
    """
    splitting the dataset into 15% test and 85% for training and validation
    """
    X = df[numerical_features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """
    Traing a lostic regression model on the training set by using
    5-fold cross validation to evaluate the model performance
    and avoid overfitting
    """
    model = LogisticRegression()
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    model.fit(X_train, y_train)
    return model, cv_scores


def train_descision_tree(X_train, y_train):
    """
    Traing a decision tree model on the training set by using
    5-fold cross validation to evaluate the model performance
    and avoid overfitting
    """
    model = DecisionTreeClassifier()
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    model.fit(X_train, y_train)
    return model, cv_scores


def evaluate_accuracy_score(y_true, y_pred):
    """
    Final evaluation on the testing set using the same method used in training (accuracy score)
    """
    return accuracy_score(y_true, y_pred)
