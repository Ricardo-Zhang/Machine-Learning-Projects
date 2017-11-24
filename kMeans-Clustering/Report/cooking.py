import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import datasets, cross_validation, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB

def index_ingredients(x, ingredients):
    v = [0]*len(ingredients)
    for i in x:
        v[ingredients.index(i)] = 1
    return v

def ingredientsList(dataFrame,ingredients):
    for data in dataFrame.ingredients:
        for ingre in data:
            ingredients.append(ingre)
    ingredients = list(np.unique(ingredients))
    return ingredients

def feature_extraction(dataFrame,ingredients):
    features = []
    for data in dataFrame.ingredients:
        features.append(index_ingredients(data, ingredients))
    return features

def label_extraction(dataFrame,cuisineList):
    labels = []
    for data in dataFrame.cuisine:
        labels.append(cuisineList.index(data))
    return labels

def main():
    train_data = pd.read_json('train.json')
    test_data = pd.read_json('test.json')
    print "Sample number: ",train_data.count()[0]
    ingredients = ingredientsList(train_data,[])
    print "Ingredient number in training set: ", len(ingredients)
    ingredients = ingredientsList(test_data,ingredients)
    print "Ingredient number: ", len(ingredients)
    cuisineList = list(np.unique(train_data.cuisine))
    print "Category number: ",len(cuisineList)
    train_features = feature_extraction(train_data,ingredients)
    train_labels = label_extraction(train_data,cuisineList)
    test_features = feature_extraction(test_data,ingredients)
    GNB = GaussianNB()
    BNB = BernoulliNB()
    log = LogisticRegression()
    print 'GaussianNB Start-------------------------------------'
    scores1 = cross_validation.cross_val_score(GNB, train_features, train_labels,cv=3)
    print 'BernoulliNB Start-------------------------------------'
    scores2 = cross_validation.cross_val_score(BNB, train_features, train_labels,cv=3)
    print 'logistic Start-------------------------------------'
    scores3 = cross_validation.cross_val_score(log, train_features, train_labels,cv=3)
    print scores1, scores2, scores3
    print np.mean(scores1), np.mean(scores2), np.mean(scores3)
    logistic = LogisticRegression()
    logistic.fit(train_features,train_labels)
    test_labels = logistic.predict(test_features)
    test_cuisines = []
    for label in test_labels:
        test_cuisines.append(cuisineList[label])
    with open('submission.csv','w') as csv_file:
        fields = ['id', 'cuisine']
        writer = csv.DictWriter(csv_file,fieldnames=fields)
        writer.writeheader()
        for i in range(len(test_cuisines)):
            writer.writerow({'id':test_data.iloc[i,0], 'cuisine':test_cuisines[i]})
    csv_file.close()

if __name__ == "__main__":
    main()
