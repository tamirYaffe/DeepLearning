import time
from PIL import Image
from numpy import asarray
import os

path_separator = os.path.sep


def load_image(path):
    image = Image.open(path)
    data = asarray(image)
    return data


def get_image_data(name, num):
    digits_to_add = 4 - len(num)
    for i in range(0, digits_to_add):
        num = '0' + num
    image_path = "dataset" + path_separator + "lfw2" + path_separator + name\
                 + path_separator + name + "_" + num + ".jpg"
    image_data = load_image(image_path)
    return image_data


def load_dataset(dataset_type):
    dataset = []
    file_path = "dataset" + path_separator + dataset_type + ".txt"
    file = open(file_path, "r")
    file.readline()
    for line in file:
        line = line.split()
        x = []
        y = 0
        if len(line) == 3:
            x.append(get_image_data(name=line[0], num=line[1]))
            x.append(get_image_data(name=line[0], num=line[2]))
            y = 1
        elif len(line) == 4:
            x.append(get_image_data(name=line[0], num=line[1]))
            x.append(get_image_data(name=line[2], num=line[3]))
        data_line = {'X': x, 'Y': y}
        dataset.append(data_line)
    file.close()
    return dataset


def main():
    train = load_dataset(dataset_type="train")
    test = load_dataset(dataset_type="test")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
