import KNN
import numpy as np
import csv
import multiprocessing 


def read_data(file_name):
  lst = []
  with open(file_name, 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader, None)
    for data in reader:
      pixels = KNN.normalizing (np.array(data, dtype='float') )
      lst.append(pixels)

  return lst 

def wraper (train_set, test_set, k, start_idx, end_idx, return_dic):
  for i in range(start_idx, end_idx + 1):
    label = KNN.knn(train_set, test_set[i], k)
    return_dic[i + 1] = label

if __name__ == "__main__":
  test_file_name = 'test.csv'
  train_file_name = 'train.csv'
  train_set = KNN.read_data(train_file_name)
  test_set = read_data(test_file_name)

  file = open('submission.csv','w')
  file.write("ImageId,Label\n")
  test_len = len(test_set)

  manager = multiprocessing.Manager()
  return_dic = manager.dict()
  GRID_hx = manager.list()
  jobs = []
  cores = 4
  segment = int(test_len / cores)

  for j in range(cores):
    start_idx = j * segment
    end_idx = (j + 1) * segment - 1
    p = multiprocessing.Process(target = wraper, args = (train_set, test_set, 5, start_idx, end_idx,return_dic))
    jobs.append(p)
    p.start()
  for j in jobs:
    j.join() 

  for i in range(test_len):
    res = str(i + 1) + ',' + str(return_dic[i + 1]) + '\n'
    file.write(res)

  file.close()


