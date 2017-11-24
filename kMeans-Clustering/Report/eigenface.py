import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm
import multiprocessing 
import sklearn.linear_model as lm
from sklearn.naive_bayes import GaussianNB
import sys


def load_data(file_name):
  labels, data = [], []
  for line in open(file_name):
      im = misc.imread(line.strip().split()[0])
      data.append(im.reshape(2500,))
      labels.append(line.strip().split()[1])
  data, labels = np.array(data, dtype=float), np.array(labels, dtype=int)
  plt.imshow(data[10, :].reshape(50,50), cmap = cm.Greys_r)
  save_path = './pic/train.jpg'
  if 'train' not in file_name:
    save_path = './pic/test.jpg'
  plt.savefig(save_path)
  plt.cla()
  return data, labels

def mean_data(save_path, data):
    size = len(data)
    ave_data = np.zeros((2500,), dtype=np.float)
    for i in range (size):
      img = data[i]
      for j in range (len(img)):
        ave_data[j] += img[j]
    for j in range (len(img)):
      ave_data[j] /= size 
    plt.imshow(ave_data.reshape(50,50), cmap = cm.Greys_r)
    plt.savefig(save_path)
    plt.cla()
    return ave_data

def mean_subtraction(save_path, ave_data, ori_data):
  mean_subtraction_data = np.subtract(ori_data, ave_data)
  plt.imshow(mean_subtraction_data.reshape(50,50), cmap = cm.Greys_r)
  plt.savefig(save_path)
  plt.clf()
  return mean_subtraction_data

def eigen_face(train_data):
  U, s, V = np.linalg.svd(train_data, full_matrices=False)
  for i in range(10):
    plt.imshow(V[i].reshape(50,50), cmap = cm.Greys_r)
    save_path = './pic/svd_' + str(i) + '_.jpg'
    plt.savefig(save_path)
    plt.clf()

def rank_approximation(train_data):
  U, s, V = np.linalg.svd(train_data, full_matrices=False)
  appro_error = np.zeros((200,), dtype=np.float)
  s_len = len(s)
  ss = np.zeros((s_len,s_len), dtype=np.float)
  for i in range (s_len):
    ss[i][i] = s[i]
  xs = np.zeros((200,), dtype=np.float)
  ys = np.zeros((200,), dtype=np.float)
  for r in range (200):
    xr = np.dot( np.dot(U[:,: r], ss[: r,: r]), V[:r,:])
    dis = np.linalg.norm(xr - train_data)
    xs[r] = r + 1.
    ys[r] = dis

  plt.xlabel('R')  
  plt.ylabel('Distance')  
  plt.title('rank-r-approximation')  #
  plt.scatter(xs, ys)
  plt.savefig('./pic/approximation.png')
  plt.clf()
  return U, ss, V

# generate r-dimensional feature matrix 
def eigenface_feature(X, r):
  U, s, V = np.linalg.svd(train_data, full_matrices=False)
  vv = V[:r,:]
  vv = vv.T
  feature_matrix_r = np.dot(X, vv)
  return feature_matrix_r

def logistic_model (train_feature_matrix, test_feature_matrix, train_labels, test_labels):
  lr = lm.LogisticRegression(max_iter = 2000, multi_class = 'ovr')
  lr.fit(train_feature_matrix, train_labels)
  accuracy = lr.score(test_feature_matrix, test_labels)
  return accuracy

def face_recognition_graph(train_data, test_data, train_labels, test_labels):
  xs = np.zeros((200,))
  ys = np.zeros((200,), dtype=np.float)
  for r in range (1, 201):
    print (r)
    train_feature_matrix = eigenface_feature(train_data, r)
    test_feature_matrix = eigenface_feature(test_data, r)
    accuracy = logistic_model(train_feature_matrix, test_feature_matrix, train_labels, test_labels)
    xs[r - 1] = r
    ys[r - 1] = accuracy
  plt.xlabel('R')  
  plt.ylabel('Accuracy')  
  plt.title('r-dimension-face-recognition-accuracy')  
  plt.scatter(xs, ys)

  plt.savefig('./pic/logistic_face_accuracy.png')
  plt.cla()

def naive_bayes(train_data, test_data, train_labels, test_labels):
  xs = np.zeros((200,))
  ys = np.zeros((200,), dtype=np.float)
  for r in range (1, 201):
    print (r)
    train_feature_matrix = eigenface_feature(train_data, r)
    test_feature_matrix = eigenface_feature(test_data, r)
    gnb = GaussianNB()
    y_pred = gnb.fit(train_feature_matrix, train_labels).predict(test_feature_matrix)
    accuracy = ((test_labels == y_pred).sum()) / float(len(test_labels))
    xs[r - 1] = r
    ys[r - 1] = accuracy

  plt.xlabel('R')  
  plt.ylabel('Accuracy')  
  plt.title('r-dimension-face-recognition-accuracy')  
  plt.scatter(xs, ys)
  plt.savefig('./pic/naiveBayes_face_accuracy.png')
  plt.cla()

if __name__ == "__main__":
  train_file_name = './faces/train.txt'
  test_file_name = './faces/test.txt'
  mean_data_train_name = './pic/mean_train.jpg'
  mean_data_test_name = './pic/mean_test.jpg'
  mean_subtraction_train_name = './pic/mean_subtraction_train.jpg'
  mean_subtraction_test_name = './pic/mean_subtraction_test.jpg'

  train_data, train_labels  = load_data(train_file_name)
  test_data, test_labels = load_data (test_file_name)
  ave_train_data = mean_data(mean_data_train_name, train_data)
  ave_test_data = mean_data(mean_data_test_name, test_data)
  mean_subtraction_train =  mean_subtraction(mean_subtraction_train_name, ave_train_data ,train_data[10])
  mean_subtraction_test = mean_subtraction(mean_subtraction_test_name, ave_test_data ,test_data[10])

  eigen_face(train_data)
  rank_approximation(train_data)
  #face_recognition_graph(train_data, test_data, train_labels, test_labels)
  naive_bayes(train_data, test_data, train_labels, test_labels)










