import os
import torch
import torch.nn as nn
import numpy as np

from demoloader.train import *
from demoloader.DCGAN import *
from utils.define_models import *
from demoloader.dataloader import *

from skimage.color import label2rgb,gray2rgb,rgb2gray
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from multiprocessing import Pool
    
from skimage.segmentation import slic



def prepare_dataset(dataset, attr, root):
    num_classes, dataset, target_model, shadow_model = get_model_dataset(dataset, attr=attr, root=root)
    length = len(dataset)
    each_length = length//4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(dataset, [each_length, each_length, each_length, each_length, len(dataset)-(each_length*4)])
    
    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model


def get_model_dataset(dataset_name, attr, root):
    if dataset_name.lower() == "utkface":
        if isinstance(attr, list):
            num_classes = []
            for a in attr:
                if a == "age":
                    num_classes.append(117)
                elif a == "gender":
                    num_classes.append(2)
                elif a == "race":
                    num_classes.append(4)
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))
        else:
            if attr == "age":
                num_classes = 117
            elif attr == "gender":
                num_classes = 2
            elif attr == "race":
                num_classes = 4
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = UTKFaceDataset(root=root, attr=attr, transform=transform)
        input_channel = 3
        
    elif dataset_name.lower() == "celeba":
        if isinstance(attr, list):
            for a in attr:
                if a != "attr":
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))

                num_classes = [8, 4]
                # heavyMakeup MouthSlightlyOpen Smiling, Male Young
                attr_list = [[18, 21, 31], [20, 39]]
        else:
            if attr == "attr":
                num_classes = 8
                attr_list = [[18, 21, 31]]
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = CelebA(root=root, attr_list=attr_list, target_type=attr, transform=transform)
        input_channel = 3

    elif dataset_name.lower() == "stl10":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.STL10(
                root=root, split='train', transform=transform, download=True)
            
        test_set = torchvision.datasets.STL10(
                root=root, split='test', transform=transform, download=True)

        dataset = train_set + test_set
        input_channel = 3

    elif dataset_name.lower() == "fmnist":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = torchvision.datasets.FashionMNIST(
                root=root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(
                root=root, train=False, download=True, transform=transform)

        dataset = train_set + test_set
        input_channel = 1

    if isinstance(num_classes, int):
        target_model = CNN(input_channel=input_channel, num_classes=num_classes)
        shadow_model = CNN(input_channel=input_channel, num_classes=num_classes)
    else:
        target_model = CNN(input_channel=input_channel, num_classes=num_classes[0])
        shadow_model = CNN(input_channel=input_channel, num_classes=num_classes[0])


    return num_classes, dataset, target_model, shadow_model


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



def multi_lime(start_index,end_index,target_train,target_model,device):

    def import_target_model():
        my_path = "/LIL/demoloader/trained_model/celeba_target.pth"        
        my_target_model = target_model.to(device)
        my_target_model.load_state_dict(torch.load(my_path))          
        return my_target_model



    def predict(input):
        # input: numpy array, (batches, height, width, channels)  
                                                                                                                                                        
        my_target_model = import_target_model()
        my_target_model.eval()

        input = torch.FloatTensor(input).permute(0, 3, 1, 2)                                                                                                                                                                                                              
        # pytorch tensor, (batches, channels, height, width)

        output = my_target_model(input.cuda())                                                                                                                                             
        return output.detach().cpu().numpy()                                                                                                                              
                                                                                                                                                                                



    explained_images = []
    explained_labels = []
    
    for index in range(start_index,end_index):
        print(index)
        image, label = target_train.__getitem__(index)   
        if len(label)>1:
            label = label[0]
        label = label.item()
  
        X_vec = image.permute(1, 2, 0).numpy().astype(np.double)          
        

        explainer = lime_image.LimeImageExplainer()             

        segmenter = SegmentationAlgorithm('slic', n_segments=20, compactness=1, sigma=1)  

        explaination = explainer.explain_instance(image=X_vec, classifier_fn=predict, 
            segmentation_fn=segmenter, 
            hide_color=0, top_labels=5,num_samples=100)
            

        lime_img, mask = explaination.get_image_and_mask(label=label, positive_only=True, negative_only=False, hide_rest=True)

                
        explained_images.append(lime_img) 
        explained_labels.append(label)

    dataset_name = "/LIL/demoloader/trained_model/explained_celeba_train_dataset_"+str(start_index)+"_"+str(end_index)
    np.save(dataset_name+'.images',np.array(explained_images))
    np.save(dataset_name+'.labels',np.array(explained_labels))



def multip(num_multiprocessing,gap,target_train,target_model,device):
    p = Pool(num_multiprocessing)
    for i in range(num_multiprocessing):
        p.apply_async(multi_lime, 
        args=(gap*i, gap*(i+1)-1,target_train,target_model,device,)
        )
    p.close()
    p.join()



if __name__=='__main__':    

    class args:
        gpu = "0"
        attributes = "attr_attr"
        dataset_name = "celeba"
        train_model = False
        data_distillation = True
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
    

    torch.save(target_train, TARGET_PATH + '_{}_target_train_dataset.pt'.format(args.attack_type))
    torch.save(target_test, TARGET_PATH + '_{}_target_test_dataset.pt'.format(args.attack_type))
    torch.save(shadow_train, TARGET_PATH + '_{}_shadow_train_dataset.pt'.format(args.attack_type))
    torch.save(shadow_test, TARGET_PATH + '_{}_shadow_test_dataset.pt'.format(args.attack_type))
  

    if args.train_model:
        start = time.time()
        train_model(TARGET_PATH, device, target_train, target_test, target_model, use_DP, noise, norm, delta)
        end = time.time()
        print("Target training time:", end-start)
        
    if args.data_distillation:    
       
        num_multiprocessing = 6
        gap = len(target_train) // num_multiprocessing

        print('Start Lime! \n\n')
        multip(num_multiprocessing,gap,target_train,target_model,device)

