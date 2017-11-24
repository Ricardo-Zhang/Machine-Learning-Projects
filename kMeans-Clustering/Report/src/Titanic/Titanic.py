import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, cross_validation, preprocessing, metrics, linear_model

def preprocess_data(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['Name','Ticket','Cabin'],axis=1)
    processed_df['Age'] = processed_df['Age'].fillna(processed_df['Age'].median())
    processed_df['Fare'] = processed_df['Age'].fillna(0)
    processed_df['Embarked'] = processed_df['Embarked'].fillna(0)
    processed_df.dropna
    print processed_df.count()
    return processed_df

def training(fold):
    train_ind, test_ind = cross_validation.train_test_split(range(len(X)))
    X_train = X[train_ind]
    X_test = X[test_ind]
    Y_train = Y[train_ind]
    Y_test = Y[test_ind]

    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(X_train, Y_train)
    return logistic.score(X_test, Y_test)

if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    processed_train = preprocess_data(train)
    X = processed_train.drop('Survived', axis = 1).values
    Y = processed_train['Survived'].values

    list_of_folds = list(cross_validation.KFold(len(X), n_folds=5))

    output = map(training, list_of_folds)
    print (sum(output))/len(output)
