import pandas as pd
from matplotlib import pyplot as plt
import numpy
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import sys

my_data = pd.read_csv('iris.data.txt', sep = ',', header = None)
for row in my_data.iterrows:
    print "Line contains: " + row

res = my_data[4].values
colors = []
features = ['sepal_length', 'sepal_width','petal_length','petal_width']
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-en']
for i in range (len(res)):
    if res[i] == classes[0]:
        colors.append("r")
    elif res[i] == classes[1]:
        colors.append("g")
    else:
        colors.append("b")

image_names = []



for i in range (4):
    for j in range (4):
        if i == j:
            xs = []
            ys = []
            plt.scatter(xs, ys, c=colors)
            plt.savefig( features[i] + ".jpg")
            img = Image.open(features[i] + ".jpg")
            d = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", 48)
            d.text((240, 140), features[i], fill=(0, 0, 0), font = font)
            img.save(features[i] + '.jpg')
            image_names.append(features[i] + ".jpg")
            continue
        xs = my_data[i].values
        ys = my_data[j].values
        plt.scatter(xs, ys, c=colors)
        plt.savefig( features[i] + "_VS_"+ features[j] + ".jpg")
        image_names.append(features[i] + "_VS_"+ features[j] + ".jpg")
        plt.cla()

images = [PIL.Image.open(i) for i in image_names ]
widths, heights = zip(*(i.size for i in images))
widths_4 = [0,0,0,0]
heights_4 = [0,0,0,0]
for i in range(16):
  widths_4[ int(i / 4) ] += widths[i]
  heights_4[i % 4] += heights[i]
max_width = max(widths_4)
max_height = max(heights_4)


new_im = Image.new('RGB', (max_width + 100, max_height + 100), color = (255,255,255))
current_height = 0
for i in range(4):
  current_max_height = 0
  x_offset = 0
  for j in range(4):
    im = images[i * 4 + j]
    new_im.paste(im, (x_offset,current_height))
    x_offset += im.size[0]
    current_max_height = max(current_max_height, im.size[1])
  current_height += current_max_height

new_im.save('final.jpg')
