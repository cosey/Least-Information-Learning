import shutil
import init_sequenc
import os
import cv2
import Mutual_information
import pickle
import sys
from multiprocessing import Pool
#path = "./Datasets/MNIST/train/"
def read_directory(directory_name):
    array_of_img = []
    for filename in os.listdir(directory_name):
        img = cv2.imread(directory_name + "/" + filename)
        #img=cv2.resize(img, (300,300), interpolation=cv2.INTER_AREA)
        array_of_img.append(img)

    return array_of_img


def task(file_number):
    path="/cifar10/imag/train/"
    file_names=os.listdir("/cifar10/imag/train/")

    #i=sys.argv[-1]
    i=file_names[file_number]
    # i=file_names[int(sys.argv[-1])]
    nn=i
    img_list = read_directory(path+str(i))

    img_number = len(img_list)
    print(img_number)
    # print(img_list[0])
    # img_number = 6

    weight_matrix = [[1]*img_number for _ in range(img_number)]
    # print(weight_matrix)

    for i in range(img_number):
        for j in range(i+1, img_number):
            try:
                MI_ij = Mutual_information.mutual_information(img_list[i], img_list[j])
                weight_matrix[i][j] = MI_ij
                weight_matrix[j][i] = MI_ij
            except:
                print(nn+str(i)+' '+str(j))
                continue
        print(str(i)+"/"+str(img_number))
                # print(weight_matrix)

    f = open('/cifar10_weight/weight_matrix_'+str(nn), 'wb')
    print('/cifar10_weight/weight_matrix_'+str(nn))
    pickle.dump(weight_matrix, f)
    f.close()

if __name__ == '__main__':

    p = Pool(10)

    for i in range(10):
        p.apply_async(task,args=(i,))

    p.close()
    p.join()


