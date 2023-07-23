import warnings
warnings.filterwarnings('ignore')
from keras.datasets import mnist, cifar10
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import numpy as np

from art import config
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import ProjectedGradientDescent
from art.utils import get_file

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32')
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


path='/models/normal_model_cifar-10.h5'
model = load_model(path)
classifier = KerasClassifier(model=model, use_logits=False, clip_values=[0,255])


y_pred = classifier.predict(X_test)
accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
print("Model Accuracy on benign test samples: %f" % (accuracy))



path_r = '/models/LIL_model_cifar-10.h5'
model_r = load_model(path_r)
classifier_r = KerasClassifier(model=model_r, use_logits=False, clip_values=[0,255])

y_pred_r = classifier_r.predict(X_test)
accuracy_r = np.mean(np.argmax(y_pred_r, axis=1) == np.argmax(y_test, axis=1))
print("Model_LIL Accuracy on benign test samples: %f" % (accuracy_r))

acc = []

for eps in [0.1*j for j in range(10)]+[0.5*i for i in range(2,21)]:

    attack = ProjectedGradientDescent(classifier, eps=eps, eps_step=1, max_iter=10, targeted=False, 
                                    num_random_init=True) 

    n = 1000
    X_test_adv = attack.generate(X_test[:n], y=y_test[:n])
    y_adv_pred = classifier.predict(X_test_adv)
    accuracy = np.mean(np.argmax(y_adv_pred, axis=1) == np.argmax(y_test[:n], axis=1))

    if accuracy==0: break

    y_adv_pred_r = classifier_r.predict(X_test_adv)
    accuracy_r = np.mean(np.argmax(y_adv_pred_r, axis=1) == np.argmax(y_test[:n], axis=1))

    acc.append((eps,accuracy,accuracy_r))

np.save(acc,'acc.txt')