import numpy as np
import csv
import matplotlib.pyplot as plt
import pylab
from scipy.spatial import distance
import copy
# Cite: https://stackoverflow.com/questions/37228371/visualize-mnist-dataset-using-opencv-or-matplotlib-pyplot
#



def display_digit(digit_sample):
    with open('train.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)

        count = 0
        for data in reader:
            label = int(data[0])
            pixels = np.array(data[1:], dtype='uint8')

            if(label in digit_sample.keys()):
                count += 1
                continue
            else:
                digit_sample[label] = [pixels, count]
                pixels = pixels.reshape((28, 28))
                plt.title('Label is {label}'.format(label=label))
                plt.imshow(pixels, cmap='gray')
                plt.savefig(str(label) + '.png')
            count += 1
            if len(digit_sample) == 10:
                break

def prior_probability():
    count = 0.0
    x = []
    with open('train.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for data in reader:
            count += 1.0
            label = int(data[0])
            x.append(label)
    num_bins = 10
    plt.clf()
    n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor='blue', alpha=1)
    plt.savefig("probability.png")

def find_nearest_neighbor(digit_sample):
    best_matches = {}
    for key in digit_sample.keys():
        best_matches[key] = [digit_sample[0], -1, key]
    count = 0
    with open('train.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for data in reader:
            label = int(data[0])
            pixels = np.array(data[1:], dtype='uint8')
            img_pixels = digit_sample[label][0]
            img_count = digit_sample[label][1]
            for key in digit_sample.keys():
                if (digit_sample[key][1] == count):
                    count += 1
                    continue
                dis = np.linalg.norm(pixels - digit_sample[key][0])
                if(best_matches[key][1] < 0 or best_matches[key][1] > dis):
                    best_matches[key] = [pixels, dis, label]
            count += 1

    for key in best_matches.keys():
        pixels = best_matches[key][0].reshape((28, 28))
        my_label = best_matches[key][2]
        plt.title('Label is {label}'.format(label=my_label))
        plt.imshow(pixels, cmap='gray')
        file_name = "bestMatch_" + str(key)
        if (key != my_label):
            file_name = file_name + '*'
        plt.savefig(file_name + '_.png')



if __name__ == "__main__":
    digit_sample = {}
    display_digit(digit_sample)
    prior_probability()
    find_nearest_neighbor(digit_sample)
