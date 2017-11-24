import numpy as np
import csv
import copy
import math
import operator
import pylab
import matplotlib.pyplot as plt
from scipy.spatial import distance
from random import randint
from multiprocessing.dummy import Pool as ThreadPool 
import multiprocessing 
import sys


# KNN classifier without external library to classify digit 


def normalizing (array):
  my_sum = 0
  m = len(array)
  for i in range(m):
    if(array[i] != 0):
      my_sum += array[i] ** 2
  tmp = float( math.sqrt(my_sum) )
  for i in range(m):
    t = float(array[i] / tmp)
    array[i] = t
  return array

def knn (training_set, test_sample, k = 3):
  dists = []
  length = (test_sample) - 1
  for data in training_set:
    pixels = data[1]
    label = data[0]
    dist = np.linalg.norm(pixels - test_sample)
    dists.append((pixels, dist, label))
  dists = sorted(dists, key=lambda x: x[1])
  neighbor_weights = {}

  for i in range(k):
    if dists[i][2] in neighbor_weights.keys():
      neighbor_weights[dists[i][2]] += (k - i) 
    else:
      neighbor_weights[dists[i][2]] = (k - i)

  curr_weight = -1
  ret_label = -1
  for label in neighbor_weights:
    if(neighbor_weights[label] > curr_weight):
      curr_weight = neighbor_weights [label]
      ret_label = label
  return ret_label



def cross_validatoin_helper(new_training_set, new_testing_set, i, res):
  true_count = 0.0
  array = np.zeros((10, 10))

  len_test = len(new_testing_set)
  for m in range(len_test):
    test_sample = new_testing_set[m][1]
    test_label = new_testing_set[m][0]
    ret_label = knn (new_training_set, test_sample, k)

    # first dimension is true label, second is classified label
    array[test_label][ret_label] += 1
    if(ret_label == test_label):
      true_count += 1.0

  print ("Print the confusion matrix, \nfirst dimension is true label\n, second is classified label\n")
  print (array)
  res_tmp = (true_count / float(len_test) )
  res[i] = res_tmp


def cross_validation(train_set, k, cross_fold):
  tot_accuracy = 0.0
  tot_len = len(train_set)
  seg_len = int( len(train_set) / cross_fold ) 

  args = []
  for i in range(cross_fold):
    new_training_set = []
    new_testing_set = []
    true_count = 0.0

    for j in range (tot_len):
      if(int(j / seg_len) == i):
        new_testing_set.append(train_set[j])
      else:
        new_training_set.append(train_set[j])
    args.append([new_training_set, new_testing_set])

  manager = multiprocessing.Manager()
  return_dic = manager.dict()
  GRID_hx = manager.list()
  jobs = []
  results = 0

  for i in range(3):
    p = multiprocessing.Process(target = cross_validatoin_helper, args = (args[i][0], args[i][1], i, return_dic))
    jobs.append(p)
    p.start()
  for j in jobs:
    j.join() 

  for j in range(3):
    print ("job " + str(j) + " with " + str(return_dic[j]))
    tot_accuracy += return_dic[j]
  res = tot_accuracy / cross_fold  
  return res

def read_data(file_name):
  lst = []
  with open(file_name, 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader, None)
    for data in reader:
      label = int(data[0])
      pixels = normalizing ( np.array(data[1:], dtype='float') )
      lst.append([label, pixels])
  return lst 

if __name__ == "__main__":
  train_file_name = 'train.csv'
  train_set = read_data(train_file_name)
  k = 5
  cross_fold = 3
  avg_accuracy = cross_validation(train_set, k, cross_fold)
  print ("The final res is " + str(avg_accuracy))



