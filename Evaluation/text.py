import os
from unittest import TestResult
import torch
import torch.nn as nn
import torchvision.models as models

from attack.meminf import *
from attack.modinv import *
from attack.attrinf import *
from demoloader.train import *
from demoloader.DCGAN import *
from utils.define_models import *
from demoloader.dataloader import *

from skimage.color import label2rgb,gray2rgb,rgb2gray
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from multiprocessing import Pool
    
from skimage.segmentation import slic
from PIL import Image
import matplotlib.pyplot as plt
import shutil

def train_model(PATH, device, train_set, test_set, model, use_DP, noise, norm, delta):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=4)
    
    model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, delta)
    acc_train = 0
    acc_test = 0

    save_res=[]
    for i in range(100):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("target training")

        acc_train, loss = model.train()
        print("target testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        print('The overfitting rate is %s' % overfitting)

        # if loss<=0.001: break
        save_res.append((i,loss))

    FILE_PATH = PATH + "_target.pth"
    model.saveModel(FILE_PATH)
    print("Saved target model!!!")
    print("Finished training!!!")

    np.save(PATH+'train_loss_record',np.array(save_res))
    return acc_train, acc_test, overfitting


def test_meminf(PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, train_shadow, use_DP, noise, norm, delta):
    batch_size = 64
    if train_shadow:
        shadow_trainloader = torch.utils.data.DataLoader(
            shadow_train, batch_size=batch_size, shuffle=True, num_workers=2)
        shadow_testloader = torch.utils.data.DataLoader(
            shadow_test, batch_size=batch_size, shuffle=True, num_workers=2)

        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        train_shadow_model(PATH, device, shadow_model, shadow_trainloader, shadow_testloader, use_DP, noise, norm, loss, optimizer, delta)
    
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(        target_train, target_test, shadow_train, shadow_test, batch_size)
    print("get_attack_dataset_with_shadow Done")

    #for white box
    gradient_size = get_gradient_size(target_model)
    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2

    attack_model = ShadowAttackModel(num_classes)

    
    attack_mode0(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, 1, num_classes)

def train_DCGAN(PATH, device, train_set, name):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=2)

    if name.lower() == 'fmnist':
        D = FashionDiscriminator(ngpu=1).eval()
        G = FashionGenerator(ngpu=1).eval()
    else:
        D = Discriminator(ngpu=1).eval()
        G = Generator(ngpu=1).eval()

    print("Starting Training DCGAN...")
    GAN = GAN_training(train_loader, D, G, device)
    for i in range(200):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        GAN.train()

    GAN.saveModel(PATH + "_discriminator.pth", PATH + "_generator.pth")


def test_modinv(PATH, device, num_classes, target_train, target_model, name):

    size = (1,) + tuple(target_train[0][0].shape)

    
    target_model, evaluation_model = load_data(PATH + "_target.pth", PATH + "_eval.pth", target_model, models.resnet18(num_classes=num_classes))


    # CCS 15
    modinv_ccs = ccs_inversion(target_model, size, num_classes, 1, 3000, 100, 0.001, 0.003, device)
    # plt.imshow(target_train[0][0].permute(1, 2, 0).numpy().astype(np.double))
    # plt.show()
    train_loader = torch.utils.data.DataLoader(target_train, batch_size=1, shuffle=False)
    ccs_result = modinv_ccs.reverse_mse(train_loader)
    print("ccs15:", ccs_result)


    # Secret Revealer
    if name.lower() == 'fmnist':
        D = FashionDiscriminator(ngpu=1).eval()
        G = FashionGenerator(ngpu=1).eval()
    else:
        D = Discriminator(ngpu=1).eval()
        G = Generator(ngpu=1).eval()

    PATH_D = PATH + "_discriminator.pth"
    PATH_G = PATH + "_generator.pth"
    
    D, G, iden = prepare_GAN(name, D, G, PATH_D, PATH_G)
    modinv_revealer = revealer_inversion(G, D, target_model, evaluation_model, iden, device)   
    print("Secret Revealer Acc:{:.6f}\t".format(modinv_revealer))

def test_attrinf(PATH, device, num_classes, target_train, target_test, target_model):

    attack_length = int(0.5 * len(target_train))
    rest = len(target_train) - attack_length

    attack_train, _ = torch.utils.data.random_split(target_train, [attack_length, rest])
    attack_test = target_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=64, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=64, shuffle=True, num_workers=2)

    image_size = [1] + list(target_train[0][0].shape)
    train_attack_model(
        PATH + "_target.pth", PATH, num_classes[1], device, target_model, attack_trainloader, attack_testloader, image_size)



def read_explained_dataset(num_multiprocessing, gap):
    explained_images =[]
    explained_labels=[]

    for i in range(num_multiprocessing):
        start_index = gap*i
        end_index = gap*(i+1)-1

        dataset_name = "/LIL/demoloader/trained_model/explained_celeba_train_dataset_"+str(start_index)+"_"+str(end_index)
        
        explained_images+=np.load(dataset_name+'.images.npy').tolist()
        explained_labels+=np.load(dataset_name+'.labels.npy').tolist()
        print(dataset_name)

    tensor_x = torch.Tensor(explained_images).permute(0,3,1,2)

    tensor_y = torch.Tensor(explained_labels).type(torch.LongTensor)



    my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) 



    return my_dataset




if __name__=='__main__':    

    class args:
        gpu = "0"
        attributes = "attr_attr"
        dataset_name = "celeba"
        attack_type = '7'
        train_model = False
        train_shadow = False
        explain_tool = 'lime'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")

    dataset_name = args.dataset_name
    attr = args.attributes
    if "_" in attr:
        attr = attr.split("_")
    root = "/LIL/data"    
    use_DP = args.use_DP
    noise = args.noise
    norm = args.norm
    delta = args.delta
    train_shadow = args.train_shadow
    TARGET_PATH = "/LIL/ML-Doctor-main/demoloader/trained_model/" + dataset_name
    TARGET_Explained_PATH="/LIL/ML-Doctor-main/demoloader/trained_model/explained_" + dataset_name


    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(dataset_name, attr, root)
    

    if args.train_model:
        start = time.time()
        train_model(TARGET_PATH, device, target_train, target_test, target_model, use_DP, noise, norm, delta)
        end = time.time()
        print("Target training time:", end-start)
        
    if args.train_LIL_model:    
       
        num_multiprocessing = 6
        gap = len(target_train) // num_multiprocessing

        explained_train_datasets = read_explained_dataset(num_multiprocessing, gap)

        print('Start Train LIL model：\n\n')
        start = time.time()
        train_model(TARGET_Explained_PATH, device, explained_train_datasets, target_test, target_model, use_DP, noise, norm, delta)
        end = time.time()
        print("LIL_Target training time:", end-start)

    if '0' in args.attack_type:
        print('start meminf attack target model')
        test_meminf(TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, train_shadow, use_DP, noise, norm, delta)        

        print('start meminf attack explained target model')
        test_meminf(TARGET_Explained_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, train_shadow, use_DP, noise, norm, delta)

    if '1' in args.attack_type:
        print('start modinv')

        print('ModInv attack: train human model!')
        target_model = models.resnet18(num_classes=num_classes)
        train_model(TARGET_PATH, device, shadow_train, shadow_test, target_model, use_DP, noise, norm, delta)

        train_DCGAN(TARGET_PATH, device, shadow_test, dataset_name)
        
        test_modinv(TARGET_PATH, device, num_classes, target_train, target_model, dataset_name)

        train_DCGAN(TARGET_Explained_PATH, device, shadow_test + shadow_train, dataset_name)
        if not os.path.exists(TARGET_Explained_PATH + "_discriminator.pth"):
            shutil.copyfile(TARGET_PATH+"_discriminator.pth", TARGET_Explained_PATH + "_discriminator.pth")
            shutil.copyfile(TARGET_PATH+"_generator.pth", TARGET_Explained_PATH + "_generator.pth")
            shutil.copyfile(TARGET_PATH+"__eval.pth", TARGET_Explained_PATH + "_eval.pth")

        test_modinv(TARGET_Explained_PATH, device, num_classes, explained_train_datasets, target_model, dataset_name)

    if '2' in args.attack_type:
        print('start attrinf')
        
        test_attrinf(TARGET_PATH, device, num_classes, target_train, target_test, target_model)

        test_attrinf(TARGET_Explained_PATH, device, num_classes, shadow_train, explained_train_datasets, target_model)