import KNN
import numpy as np
import csv
import matplotlib.pyplot as plt
import multiprocessing 
from statsmodels.distributions.empirical_distribution import ECDF
# For problem 1.d and 1.e

def read_data(file_name):
  lst = []
  with open(file_name, 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader, None)
    #idx = 0
    for data in reader:
      label = int(data[0])
      if label == 0 or label == 1:
        #idx += 1
        #if idx >= 51:
          break
        pixels = KNN.normalizing( np.array(data[1:], dtype='float') )
        lst.append([label, pixels])
  return lst 

def wrape_dist (train_set, start_idx, end_idx, dic):
  train_len = len(train_set)
  num = 0
  lst0 = []
  lst1 = []
  for i in range(start_idx, end_idx + 1):
    for j in range(i + 1, train_len):
      dist = np.linalg.norm(train_set[i][1] - train_set[j][1])
      if(train_set[i][0] == train_set[j][0]):
        lst0.append(dist)
      else:
        lst1.append(dist)
  dic[0].append(lst0)
  dic[1].append(lst1)

  print ("len dic[0] " + str(len(lst0)) + ' len dic[1] ' + str(len(lst1)))

# First is genuine distances and second is impostor distances
def pair_distance(train_set):
  train_len = len(train_set)
  dists = [[], []]

  manager = multiprocessing.Manager()
  return_dic = manager.dict()
  lst0 = manager.list()
  lst1 = manager.list()
  return_dic[0] = lst0
  return_dic[1] = lst1
  jobs = []

  begin = [0,1,5,14,30]
  segment = int(train_len / 30)

  for i in range (4):
    start_idx = segment * begin[i]
    end_idx = segment * begin[i + 1] - 1
    # Make sure it reach the end
    if i == 4:
      end_idx = train_len - 1
    p = multiprocessing.Process(target = wrape_dist, args = (train_set, start_idx, end_idx ,return_dic))
    jobs.append(p)
    p.start()
  for job in jobs:
    job.join() 


  dists[0] = [item for sublist in return_dic[0] for item in sublist]
  dists[1] = [item for sublist in return_dic[1] for item in sublist]
  print ("pair_dist done")
  return dists

def plot_distances (dists):
  bins = np.linspace(0.0, 1.75, 100)
  plt.hist(dists[0], bins, alpha=0.5, label='genuine distance')
  plt.hist(dists[1], bins, alpha=0.5, label='impostor distance')
  plt.legend(loc='upper right')
  plt.savefig('roc.png')


def wrape_write(lst, file_name):
  file = open(file_name,'w')
  for data in lst:
    res = str(data) + '\n'
    file.write(res)
  file.close()

def write_down(dists):
  manager = multiprocessing.Manager()
  GRID_hx = manager.list()
  jobs = []
  file_names = ['gen.csv', 'imposter.csv']
  for i in range(2):
    p = multiprocessing.Process(target = wrape_write, args = (dists[i], file_names[i]))
    jobs.append(p)
    p.start()
  for job in jobs:
    job.join()


if __name__ == "__main__":
  train_file_name = 'train.csv'
  train_set = read_data(train_file_name)
  dists = pair_distance(train_set)

  #write the file down for to avoid later unnecessary computation
  write_down(dists)

  plot_distances(dists)
  #draw_roc(dists)