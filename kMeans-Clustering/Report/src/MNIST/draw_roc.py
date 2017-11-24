import KNN
import numpy as np
import csv
import matplotlib.pyplot as plt
import multiprocessing 
from statsmodels.distributions.empirical_distribution import ECDF

def read_data(file_name):
  lst = []
  with open(file_name, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for data in reader:
      lst.append(float(data[0]))
  return lst 


def draw_roc (gen_lst, imposter_lst):
  gen_ecdf = ECDF (gen_lst)
  impos_ecdf = ECDF (imposter_lst)

  t = np.linspace (0, max (max(gen_lst), max(imposter_lst)),10000, endpoint=True)
  gen_y = gen_ecdf(t)
  impos_y = impos_ecdf (t)
  plt.xlabel('False Positive Rate')  
  plt.ylabel('True Positive Rate')  
  plt.title('ROC Curve')  
  plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
  plt.plot(impos_y, gen_y)
  plt.savefig('roc2.png')

if __name__ == "__main__":
  gen_name = 'gen.csv'
  imposter_name = 'imposter.csv'
  gen_lst = read_data(gen_name)
  imposter_lst = read_data(imposter_name)
  draw_roc(gen_lst, imposter_lst)