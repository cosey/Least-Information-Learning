import os, sys
from os.path import abspath

class temp_model_dir():
    name = '/LIL/Poison_Attack/temp/'

import torch
import numpy as np
from art.utils import load_dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')
# Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
mean = (0.4914, 0.4822, 0.4465) 
std = (0.2023, 0.1994, 0.201)


from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import torch.optim as optim

num_classes=10
feature_size=4096
model=nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Dropout(),
        nn.Linear(256 * 1 * 1, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, feature_size),
        nn.ReLU(inplace=True),
        nn.Linear(feature_size, num_classes)
)

# Define the ART Estimator
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=2e-4)
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_, max_),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
    preprocessing=(mean, std)
)

# Train the model
classifier.fit(x_train, y_train, nb_epochs=100, batch_size=128, verbose=True)
for param_group in classifier.optimizer.param_groups:
    print(param_group["lr"])
    param_group["lr"] *= 0.1
classifier.fit(x_train, y_train, nb_epochs=50, batch_size=128, verbose=True)
for param_group in classifier.optimizer.param_groups:
    print(param_group["lr"])
    param_group["lr"] *= 0.1
classifier.fit(x_train, y_train, nb_epochs=50, batch_size=128, verbose=True)
torch.save(model.state_dict(), temp_model_dir.name + "/htbd_model.pth") # Write the checkpoint to a temporary directory

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))



from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
target = np.array([0,0,0,0,1,0,0,0,0,0])
source = np.array([0,0,0,1,0,0,0,0,0,0])

# Backdoor Trigger Parameters
patch_size = 8
x_shift = 32 - patch_size - 5
y_shift = 32 - patch_size - 5

# Define the backdoor poisoning object. Calling backdoor.poison(x) will insert the trigger into x.
from art.attacks.poisoning import perturbations
def mod(x):
    original_dtype = x.dtype
    x = perturbations.insert_image(x, backdoor_path="./temp/htbd.png",
                                   channels_first=True, random=False, x_shift=x_shift, y_shift=y_shift,
                                   size=(patch_size,patch_size), mode='RGB', blend=1)
    return x.astype(original_dtype)
backdoor = PoisoningAttackBackdoor(mod)



from art.attacks.poisoning import HiddenTriggerBackdoor
poison_attack = HiddenTriggerBackdoor(classifier, eps=16/255, target=target, source=source, feature_layer=19, backdoor=backdoor, decay_coeff = .95, decay_iter = 2000, max_iter=5000, batch_size=25, poison_percent=.01)

poison_data, poison_indices = poison_attack.poison(x_train, y_train)
print("Number of poison samples generated:", len(poison_data))


# Create finetuning dataset
dataset_size = 2500
num_classes = 10
num_per_class = dataset_size/num_classes

poison_dataset_inds = []

for i in range(num_classes):
    class_inds = np.where(np.argmax(y_train,axis=1) == i)[0]
    num_select = int(num_per_class)
    if np.argmax(target) == i:
        num_select = int(num_select - len(poison_data))
        poison_dataset_inds.append(poison_indices)
    poison_dataset_inds.append(np.random.choice(class_inds, num_select, replace=False))
    
poison_dataset_inds = np.concatenate(poison_dataset_inds)

poison_x = np.copy(x_train)
poison_x[poison_indices] = poison_data
poison_x = poison_x[poison_dataset_inds]

poison_y = np.copy(y_train)[poison_dataset_inds]


import torch.nn as nn

# Load model again
num_classes=10
feature_size=4096
model=nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Dropout(),
        nn.Linear(256 * 1 * 1, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, feature_size),
        nn.ReLU(inplace=True),
        nn.Linear(feature_size, num_classes)
)
model.load_state_dict(torch.load(temp_model_dir.name+"/htbd_model.pth"))
# temp_model_dir.cleanup() # Remove the temporary directory after loading the checkpoint

# Freeze the layers up to the last layer
for i, param in enumerate(model.parameters()):
    param.requires_grad = False


num_classes=10
feature_size=4096
model[20] = nn.Linear(feature_size, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=2e-4)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_, max_),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
    preprocessing=(mean, std)
)


trigger_test_inds = np.where(np.all(y_test == source, axis=1))[0]

lr_factor = .1
lr_schedule = [5, 10, 15]

test_poisoned_samples, test_poisoned_labels  = backdoor.poison(x_test[trigger_test_inds], y_test[trigger_test_inds])

for i in range(4):
    print("Training Epoch", i*5)
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    
    predictions = classifier.predict(x_test[trigger_test_inds])
    b_accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[trigger_test_inds], axis=1)) / len(trigger_test_inds)
    
    predictions = classifier.predict(test_poisoned_samples)
    p_accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(test_poisoned_labels,axis=1)) / len(test_poisoned_labels)
    p_success = np.sum(np.argmax(predictions, axis=1) == np.argmax(target)) / len(test_poisoned_labels)
    if i != 0:
        for param_group in classifier.optimizer.param_groups:
            param_group["lr"] *= lr_factor
    classifier.fit(poison_x, poison_y, epochs=5, training_mode=False)


print("Final Performance")
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

predictions = classifier.predict(x_test[trigger_test_inds])
b_accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[trigger_test_inds], axis=1)) / len(trigger_test_inds)
print("Accuracy on benign trigger test examples: {}%".format(b_accuracy * 100))

predictions = classifier.predict(test_poisoned_samples)
p_accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[trigger_test_inds],axis=1)) / len(trigger_test_inds)
print("Accuracy on poison trigger test examples: {}%".format(p_accuracy * 100))
p_success = np.sum(np.argmax(predictions, axis=1) == np.argmax(target)) / len(trigger_test_inds)
print("Success on poison trigger test examples: {}%".format(p_success * 100))