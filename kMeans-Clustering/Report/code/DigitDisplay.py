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
