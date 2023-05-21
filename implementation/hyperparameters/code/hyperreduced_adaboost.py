# Help from: https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
# Code from Explainable AI with Python

# True to print debugging outputs, False to silence the program
DEBUG = True
separator = "-------------------------------------------------------------------------"
# Correlation threshold for Pearson correlation. For feature pairs with correlation higher than the threshold, one feature is dropped 
CORRELATION_THRESHOLD = 0.9

# Import the necessary libraries (tested for Python 3.9)
import numpy as np
import pandas as pd
import os
from sklearn import ensemble
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import sys

# Dataset to load
filename = "./labeled_dataset_features.csv"

file_out = "./grid_search_results_adaboost.csv"
fdw = open(file_out, "w")

def load_dataset(filename):
    # Load the dataset in the form of a csv
    df = pd.read_csv(filename)
    headers = pd.read_csv(filename, index_col = False, nrows = 0).columns.tolist()
    features = headers[0:-3]

    # Return a dataframe and the names of the features
    return df, features

def drop_features_by_correlation(df):
    # Calculate correlation coefficients for pairs of features
    df_for_corr = df.drop(labels = ['Name', 'Label', 'Family'], axis = 1)
    correlation_coeffs = df_for_corr.corr()

    # Keep the upper triangular matrix of correlation coefficients
    upper_tri = correlation_coeffs.where(np.triu(np.ones(correlation_coeffs.shape), k = 1).astype(np.bool))

    # Drop columns with high correlation (one of the features consisting the pair is dropped, the other is kept)
    to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) >= CORRELATION_THRESHOLD)]

    if DEBUG == True:
        print("Correlation threshold is:")
        print(CORRELATION_THRESHOLD)
        print(separator)

    df = df.drop(columns = to_drop, inplace = False)
    features = df.columns.tolist()[0:-3]

    # Return the names of the dropped features, the new dataframe and the names of the new features within the feature set
    return to_drop, df, features

def split_dataset(df):
    # Split the dataset into training and testing sets
    train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 2345, shuffle = True)

    # Split features from labels (the last three columns are domain name, binary label, malware family)
    X_train = train_set.iloc[:, :-3]
    y_train = train_set.iloc[:, -2:-1]
    X_test = test_set.iloc[:, :-3]
    y_test = test_set.iloc[:, -3:]

    return X_train, y_train, X_test, y_test

def scale_dataset(X_train, X_test):
    # Scale the dataset using min-max scaling
    minimum = X_train.min()
    maximum = X_train.max()
    X_train = (X_train - minimum) / (maximum - minimum)
    X_test = (X_test - minimum) / (maximum - minimum)

    # Return the scaled training and testing datasets
    return X_train, X_test

def oversample_data(X_train, y_train):
    # Oversample the data using SMOTE
    sm = SMOTE(random_state = 42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, y_train

def train_and_test_model(X_train, y_train, X_test, y_test):
    rng = np.random.RandomState(42)

    os.system("rm ./results_adaboost.csv")
    os.system("touch ./results_adaboost.csv")

    for estimators in range(10, 1010, 10):
            model = AdaBoostClassifier(n_estimators = estimators, random_state = rng)
            model.fit(X_train, y_train.values.ravel())
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test["Label"].values, predictions, normalize = True)

            if DEBUG == True:
                txt = str(accuracy) + "," + str(estimators)
                print(txt)

            to_write = str(accuracy) + "," + str(estimators)
            command = 'echo "' + str(to_write) + '" >> results_adaboost.csv'
            os.system(command)
            #fdw.write(to_write + "\n")

    return None

if __name__ == "__main__":
    # Load the dataset
    df, features = load_dataset(filename)

    if DEBUG == True:
        print("Before correlation: The dataframe is:")
        print(df)
        print(separator)
        print("Before correlation: The shape of the dataframe is:")
        print(df.shape)
        print(separator)
        print("Before correlation: The names of the features are:")
        print(features)
        print(separator)

    # Drop features based on Pearson correlation
    to_drop, df, features = drop_features_by_correlation(df)
    print("Dropped features because of correlation: ", str(to_drop))

    if DEBUG == True:
        print("After correlation: The new dataframe is:")
        print(df)
        print(separator)
        print("After correlation: The shape of the dataframe is:")
        print(df.shape)
        print(separator)
        print("After correlation: The names of the features are:")
        print(features)
        print(separator)

    # Split dataset into training and testing portions
    X_train, y_train, X_test, y_test = split_dataset(df)

    if DEBUG == True:
        print("Unscaled X_train:")
        print(X_train)
        print(separator)
        print("Size of X_train:")
        print(len(X_train))
        print(separator)
        print("y_train:")
        print(y_train)
        print(separator)
        print("Size of y_train:")
        print(len(y_train))
        print(separator)
        print("Unscaled X_test:")
        print(X_test)
        print(separator)
        print("Size of X_test:")
        print(len(X_test))
        print(separator)
        print("y_test:")
        print(y_test)
        print(separator)
        print("Size of y_test:")
        print(len(y_test))
        print(separator)

    # Scale dataset using min-max scaling
    X_train, X_test = scale_dataset(X_train, X_test)

    if DEBUG == True:
        print("Scaled X_train:")
        print(X_train)
        print(separator)
        print("Scaled X_test:")
        print(X_test)
        print(separator)

    # Data oversampling to deal with class imbalance
    X_train, y_train = oversample_data(X_train, y_train)

    if DEBUG == True:
        print("Size of oversampled X_train:")
        print(len(X_train))
        print(separator)
        print("Size of oversampled y_train:")
        print(len(y_train))
        print(separator)

    train_and_test_model(X_train, y_train, X_test, y_test)
        
fdw.close()
