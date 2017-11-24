import csv
import numpy as np
import matplotlib.pyplot as plt

def display_digit():
    with open('test.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)

        for data in reader:
            pixels = np.array(data, dtype='uint8')
            pixels = pixels.reshape((28, 28))
            plt.imshow(pixels, cmap='gray')
            plt.savefig( 'test.png')
            break

if __name__ == "__main__":
    display_digit()