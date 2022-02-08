import os

image_path = './MaskPascalVOC/images/'
labels_path = './MaskPascalVOC/images/'

images = os.listdir(image_path)
labels = os.listdir(labels_path)

split = open('./split.txt', mode='a', encoding='utf-8')
for i in os.walk(list(images)):
    f = open(labels_path+i+'.txt')
    split.write(image_path+images[i]+' ')
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        for j in range(len(line)):
            line[j] = line[j].rstrip()
        print("line :", line)
        split.write(line[1]+','+line[2]+','+line[3]+','+line[4]+line[0]+' ')

    split.write("\n")