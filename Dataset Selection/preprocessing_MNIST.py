import numpy as np
import cv2
import os

def save_mnist_to_jpg(mnist_image_file, mnist_label_file, save_dir):
    if 'train' in os.path.basename(mnist_image_file):
        num_file = 60000
        prefix = 'train'
    else:
        num_file = 10000
        prefix = 'test'
    with open(mnist_image_file, 'rb') as f1:
        image_file = f1.read()
    with open(mnist_label_file, 'rb') as f2:
        label_file = f2.read()
    image_file = image_file[16:]
    label_file = label_file[8:]

    j = [0]*10
    for i in range(num_file):
        # label = int(label_file[i].encode('hex'), 16)
        label = label_file[i]
        # image_list = [int(item.encode('hex'), 16) for item in image_file[i*784:i*784+784]]
        image_list = [item for item in image_file[i*784:i*784+784]]
        image_np = np.array(image_list, dtype=np.uint8).reshape(28,28,1)

        save_path = os.path.join(save_dir, '{}/{}/'.format(prefix, label))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_name = save_path + '{}.jpg'.format(j[label])
        cv2.imwrite(save_name, image_np)
        print ('{} ==> {}_{}_{}.jpg'.format(i, prefix, label,j[label]))
        j[label]+=1

if __name__ == '__main__':
    path = '../experiment/datasets/MNIST/raw/'
    train_image_file = path + 'train-images-idx3-ubyte'
    train_label_file = path + 'train-labels-idx1-ubyte'
    test_image_file = path + 't10k-images-idx3-ubyte'
    test_label_file = path + 't10k-labels-idx1-ubyte'

    save_train_dir = './Datasets/MNIST/'
    save_test_dir ='./Datasets/MNIST/'

    if not os.path.exists(save_train_dir):
        os.makedirs(save_train_dir)
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)

    save_mnist_to_jpg(train_image_file, train_label_file, save_train_dir)
    save_mnist_to_jpg(test_image_file, test_label_file, save_test_dir)