import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import datasets, cross_validation, preprocessing, metrics, linear_model

def preprocess_data(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['Name','Ticket','Cabin'],axis=1)
    processed_df['Age'] = processed_df['Age'].fillna(0)
    processed_df['Fare'] = processed_df['Age'].fillna(0)
    processed_df['Embarked'] = processed_df['Embarked'].fillna(0)
    processed_df.dropna
    print processed_df.count()
    return processed_df

def predict(test):
    #train_ind, test_ind = cross_validation.train_test_split(range(len(X)))

    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(X, Y)
    return logistic.predict(test)

if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    print test.count
    processed_train = preprocess_data(train)
    processed_test = preprocess_data(test)
    print processed_test['Age']
    X = processed_train.drop('Survived', axis = 1).values
    Y = processed_train['Survived'].values
    test = processed_test
    output = predict(test)
    print len(output)
    print output

    with open('submission.csv','wb') as csvfile:
        submission_writer = csv.writer(csvfile, delimiter = ',',
        quotechar='|',quoting = csv.QUOTE_MINIMAL)
        submission_writer.writerow(['PassengerId', 'Survived'])
        for i in range(len(output)):
            id = test.iloc[i]['PassengerId']
            submission_writer.writerow([int(id), output[i]])
    csvfile.close()
